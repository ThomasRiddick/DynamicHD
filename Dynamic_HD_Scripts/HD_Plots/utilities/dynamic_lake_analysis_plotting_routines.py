'''
A module containing routines to assisting in making plots for
analysing the product of dynamic lake model dry runs

Created on Jun 4, 2021

@author: thomasriddick
'''
import glob
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from HD_Plots.utilities.match_river_mouths import generate_matches
from HD_Plots.utilities import match_river_mouths

def find_highest_version_for_date(base_dir_template,date):
    split_string = base_dir_template.rsplit("VERSION_NUMBER",1)
    wildcarded_base_dir_template = "*".join(split_string)
    versions_for_date = glob.glob(wildcarded_base_dir_template)
    version_numbers = [ re.match(r"_([0-9]*)_",
                                 version_for_date.rsplit("_version",1)[1]).group(1)
                        for version_for_date in versions_for_date]
    return max(version_numbers)


def prepare_matched_catchment_numbers(catchments_one,
                                      catchments_two,
                                      river_mouths_one,
                                      river_mouths_two):
    params = match_river_mouths.Params("very_extensive")
    conflict_free_pairs, pairs_from_unresolved_conflicts = generate_matches(river_mouths_one,
                                                                            river_mouths_two,params)
    matched_catchments_one = np.zeros(catchments_one.shape)
    matched_catchments_two = np.zeros(catchments_one.shape)
    for pair in conflict_free_pairs:
        catchment_num_one = catchments_one[pair[0].get_coords()]
        catchment_num_two = catchments_two[pair[1].get_coords()]
        matched_catchments_one[catchments_one == catchment_num_one] = catchment_num_one
        matched_catchments_two[catchments_two == catchment_num_two] = catchment_num_one
    return matched_catchments_one, matched_catchments_two

def generate_catchment_and_cflow_comp_slice(colors,
                                            lsmask,
                                            glacier_mask,
                                            matched_catchment_nums_one,
                                            matched_catchment_nums_two,
                                            river_flow_one,
                                            river_flow_two,
                                            minflowcutoff,
                                            use_glacier_mask=True):
    colour_codes = np.zeros(lsmask.shape)
    colour_codes[lsmask == 0] = 1
    colour_codes[np.logical_and(matched_catchment_nums_one != 0,
                                matched_catchment_nums_one ==
                                matched_catchment_nums_two) ] = 5
    colour_codes[np.logical_and(np.logical_or(matched_catchment_nums_one != 0,
                                              matched_catchment_nums_two != 0),
                                matched_catchment_nums_one !=
                                matched_catchment_nums_two) ] = 6
    colour_codes[np.logical_and(river_flow_one >= minflowcutoff,
                                river_flow_two >= minflowcutoff)] = 2
    colour_codes[np.logical_and(river_flow_one >= minflowcutoff,
                                river_flow_two < minflowcutoff)]  = 3
    colour_codes[np.logical_and(river_flow_one < minflowcutoff,
                                river_flow_two >= minflowcutoff)] = 4
    if use_glacier_mask:
        colour_codes[glacier_mask == 1] = 7
    colour_codes[lsmask == 1] = 0
    cmap = mpl.colors.ListedColormap(['blue','peru','black','green','red','white','yellow'])
    bounds = list(range(8))
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    im = plt.imshow(colour_codes,cmap=cmap,norm=norm,interpolation="none")
    return im

def extract_zoomed_section(data_in,zoomed_section_bounds):
    return data_in[zoomed_section_bounds["min_lat"]:
                   zoomed_section_bounds["max_lat"]+1,
                   zoomed_section_bounds["min_lon"]:
                   zoomed_section_bounds["max_lon"]+1]

def generate_catchment_and_cflow_comp_sequence(colors,
                                               lsmask_sequence,
                                               glacier_mask_sequence,
                                               catchment_nums_one_sequence,
                                               catchment_nums_two_sequence,
                                               river_flow_one_sequence,
                                               river_flow_two_sequence,
                                               river_mouths_one_sequence,
                                               river_mouths_two_sequence,
                                               date_text_sequence,
                                               minflowcutoff,
                                               use_glacier_mask=True,
                                               zoomed=False,
                                               zoomed_section_bounds={}):
    im_list = []
    for (lsmask_slice,glacier_mask_slice,catchment_nums_one_slice,
         catchment_nums_two_slice,river_flow_one_slice,river_flow_two_slice,
         river_mouths_one_slice,river_mouths_two_slice,date_text) in \
            zip(lsmask_sequence,glacier_mask_sequence,
                catchment_nums_one_sequence,
                catchment_nums_two_sequence,
                river_flow_one_sequence,river_flow_two_sequence,
                river_mouths_one_sequence,river_mouths_two_sequence,
                date_text_sequence):
        if zoomed:
            lsmask_slice_zoomed=extract_zoomed_section(lsmask_slice,zoomed_section_bounds)
            glacier_mask_slice_zoomed=extract_zoomed_section(glacier_mask_slice,
                                                             zoomed_section_bounds)
            catchment_nums_one_slice_zoomed=extract_zoomed_section(catchment_nums_one_slice,
                                                                   zoomed_section_bounds)
            catchment_nums_two_slice_zoomed=extract_zoomed_section(catchment_nums_two_slice,
                                                                   zoomed_section_bounds)
            river_flow_one_slice_zoomed=extract_zoomed_section(river_flow_one_slice,
                                                               zoomed_section_bounds)
            river_flow_two_slice_zoomed=extract_zoomed_section(river_flow_two_slice,
                                                               zoomed_section_bounds)
            river_mouths_one_slice_zoomed=extract_zoomed_section(river_mouths_one_slice,
                                                                 zoomed_section_bounds)
            river_mouths_two_slice_zoomed=extract_zoomed_section(river_mouths_two_slice,
                                                                 zoomed_section_bounds)
        else:
            lsmask_slice_zoomed=lsmask_slice
            glacier_mask_slice_zoomed=glacier_mask_slice
            catchment_nums_one_slice_zoomed=catchment_nums_one_slice
            catchment_nums_two_slice_zoomed=catchment_nums_two_slice
            river_flow_one_slice_zoomed=river_flow_one_slice
            river_flow_two_slice_zoomed=river_flow_two_slice
            river_mouths_one_slice_zoomed=river_mouths_one_slice
            river_mouths_two_slice_zoomed=river_mouths_two_slice
        matched_catchment_nums_one,matched_catchment_nums_two =\
             prepare_matched_catchment_numbers(catchment_nums_one_slice_zoomed,
                                               catchment_nums_two_slice_zoomed,
                                               river_mouths_one_slice_zoomed,
                                               river_mouths_two_slice_zoomed)
        im_list.append([generate_catchment_and_cflow_comp_slice(colors,
                                                                lsmask_slice_zoomed,
                                                                glacier_mask_slice_zoomed,
                                                                matched_catchment_nums_one,
                                                                matched_catchment_nums_two,
                                                                river_flow_one_slice_zoomed,
                                                                river_flow_two_slice_zoomed,
                                                                minflowcutoff=minflowcutoff,
                                                                use_glacier_mask=
                                                                use_glacier_mask),date_text])
    return im_list
