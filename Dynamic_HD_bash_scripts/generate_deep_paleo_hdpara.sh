#!/bin/bash
# Author - Thomas Riddick (thomas.riddick@mpimet.mpg.de)
# Date - 4th November 2021 
set -e

#This is a template for how to produce a hdpara file for "deep" paleo time slice - i.e. one where present day river pathways are not useful in determining past river pathways as to much local geomorphic change will have occurred in the interim. This in effect means more than a few glacial cycles before the present.

#This script assumes you have an orography of some kind and a landsea mask for the time slice in question.

#In the absence of an orography some kind of best guess riverpath ways can be produce using just a landsea mask (or a landsea mask plus some information on where mountain ranges might be expected) - however instructions for this case are not provided here. If such instruction are required please get in touch with me.

#There are three variations of this script given - one if your orography is on a coarser grid than a half degree regular latitude-longitude grid, one if is on a (considerably) finer and one if it isthe exactly on a 10 minute regular lat-lon grid. The first case could also handle an orography that is finer than a half degree grid but not considerably so.

#This script operates by writing out the required ini (using standard ini file format - see https://en.wikipedia.org/wiki/INI_file) files to run various tools (in Fortran and C++) that are interfaced through hd_operator_driver.py. The script then runs a chain of these tools to create the necessary hdpara file.

#The basic process is:
#1) Fill any depressions in the orography using a depression filling algorithm
#2) Generate a set of river directions according to line of steepest decent (with a sub algoirthm to handle flat areas)
#3) Generate the flow parameter and create the final hdpara file
#(Alternatively step 2 can use the raw unfilled orography and instead use a river carving routine. The result will be the same outside of depressions.)

#The exact process for an orography coarser than a half degree grid is
#1) Interpolate the orography to a half degree grid (using bilinear interpolation)
#2) Fill sinks on the half degree orography created
#3) Generate river directions on a half degree grid according to line of steepest decent using the sink filled orography
#4) Generate flow parameters (again using the sink filled orography) and create the final hdpara file
#5) Produce diagnostic output for evaluation purposes from the river directions; namely, the catchments and cumulative flows.

#For an orography that is already on a half degree grid the exact process is the same but step 1 is skipped.

#For an orography that is on a grid finer than a half degree grid the exact process is:
#1) Fill sinks in the orography on its native grid
#2) Generate river directions from the sink-filled orography again on the orography's native grid
#3) Produce cumulative flows and catchments for the river directions on the orography's native grid. The latter is just for evaluation;
#   the former is required for the next step as well as being useful for evaluation.
#4) Upscale the river directions to the half degree grid using the COTAT+ algorithm
#5) Upscale the orography to a half degree grid using bilinear interpolation
#6) Generate flow parameters with the river directions from step 3 and the orography from step 4 and then generate the final hdpara file
#7) Produce diagnostic output for evaluation purposes from the river directions; namely, the catchments and cumulative flows.


#In each case the input orography should be the finest orography available. The input land-sea mask should be the landsea mask that ECHAM will use. They do NOT need to be on the same grid.
#
#For orographies finer than 0.5 degree an upscaling factor also need to be supplied

#References
#
# Sink Filling Algorithm
#
# Soille, P. and Gratin, C.: An Efficient Algorithm for Drainage Net- work Extraction on DEMs, J. Vis. Commun. Image R., 5, 181– 189, https://doi.org/10.1006/jvci.1994.1017, 1994.
# Wang, L. and Liu, H.: An efficient method for identifying and fill- ing surface depressions in digital elevation models for hydro- logic analysis and modelling, Int. J. Geogr. Inf. Sci., 20, 193– 213, https://doi.org/10.1080/13658810500433453, 2006.
#
# COTAT+ upscaling algorithm
# Paz, A. R., Collischonn, W., and Lopes da Silveira, A. L.: Improvements in large-scale drainage networks derived from digital elevation models, Water Resour. Res., 42, W08502, https://doi.org/10.1029/2005WR004544, 2006.
#
# I am not clear what the official reference for downslope routing is or if its origin is known.
#
# Catchment Computation - I don't usually reference for this. Although not the originally source of this I took the algorithm from the paper
# Barnes, R., Lehman, C., and Mulla, D.: Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models, Comput. Geosci., 62, 117–127, 2014.
# Cumulative Flow - Unclear. I also don't usually reference this. I wrote the algorithm used here myself but this idea has a existed for a long time and I am certain this way of computing it has been published in various form in several places before.
#
# Flow parameter generation - these are reference for the overall HD scheme of which the formula for flow paramters is an
# integeral part
#
# Hagemann, S. and Dümenil, L.: Documentation for the Hydro- logical Discharge Model, Tech. Rep. 17, Max Planck Institute for Meteorology, Bundesstraße 55, 20146, Hamburg, Germany, 1998a.
#Hagemann, S. and Dümenil, L.: A parametrization of the lat- eral waterflow for the global scale, Clim. Dynam., 14, 17–31, https://doi.org/10.1007/s003820050205, 1998b.

#Input parameters
is_coarser_than_half_degree=false
is_half_degree=false
is_finer_than_half_degree=true
upscaling_factor=3
no_compile=false

#Input filepaths and fieldnames
dynamic_hd_code=/path/to/dynamic/hd/release
working_directory=/path/to/working/directory
landsea_file_in=/path/to/landsea_file
landsea_fieldname="slm"
orography_file_in=/path/to/orography/file
orography_fieldname="z"
ancillary_data_path=/path/to/ancillary/data
cotat_plus_parameters_path=${ancillary_data_path}/cotat_plus_standard_params.nl

#Can only use one option at a time!! Check this is the case
if $is_coarser_than_half_degree && $is_half_degree ; then
	exit 1
fi
if $is_coarser_than_half_degree && $is_finer_than_half_degree ; then
	exit 1
fi
if $is_half_degree && $is_finer_than_half_degree; then
	exit 1
fi 

if $no_compile ; then
    no_compile_flag="-n"
else
    no_compile_flag=""
fi

#Move to working directory
cd ${working_directory}

#Make directory for parameter generation
mkdir paragen

#Select and Invert Mask


if $is_coarser_than_half_degree || $is_half_degree ; then
cat > fill_sinks.ini << EOF
[general]
    operation:sink_filling
[sink_filling]
    algorithm:direct_sink_filling
    add_slight_slope_when_filling_sinks:False
    slope_param:0.0
[input_filepaths]
    orography:orography_30min.nc
    landsea:slm_30min.nc
[input_fieldnames]
    orography:${orography_fieldname}
    landsea:slm
[output_filepaths]
    orography_out:filled_orography_30min.nc
[output_fieldnames]
    orography_out:z
EOF

cat > generate_rdirs.ini << EOF
[general]
    operation:determine_river_directions
[river_direction_determination]
    always_flow_to_sea:True
    use_diagonal_nbrs:True
    mark_pits_as_true_sinks:True
[input_filepaths]
    landsea:slm_30min.nc
    orography:filled_orography_30min.nc
[input_fieldnames]
    landsea:slm
    orography:z
[output_filepaths]
    rdirs_out:rdirs_30min.nc
[output_fieldnames]
    rdirs_out:rdirs
EOF

cat > generate_hdpara_file.ini << EOF
[general]
    operation:parameter_generation
    ancillary_data_path:${ancillary_data_path}
    working_dir:${working_directory}/paragen
[input_filepaths]
    landsea:$(pwd)/slm_30min_inverse.nc
    orography:$(pwd)/filled_orography_30min.nc
    rdirs:$(pwd)/rdirs_30min.nc
[input_fieldnames]
    landsea:slm
    orography:z
    rdirs:rdirs
[output_filepaths]
    hdpara_out:$(pwd)/final_hdpara.nc
EOF

cat > compute_catchments.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:catchment_computation
[input_filepaths]
    rdirs:rdirs_30min.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    loop_logfile:loops_30min.txt
    catchments_out:catch_30min.nc
[output_fieldnames]
    catchments_out:catch
EOF

cat > compute_cumulative_flow.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:cumulative_flow_computation
[input_filepaths]
    rdirs:rdirs_30min.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    cumulative_flow_out:acc_30min.nc
[output_fieldnames]
    cumulative_flow_out:acc
EOF

 cdo select,name=${landsea_fieldname} ${landsea_file_in} original_slm_selected.nc
 cdo chname,${landsea_fieldname},slm original_slm_selected.nc original_slm_renamed.nc
 cdo expr,"slm=(${landsea_fieldname}==0)" ${landsea_file_in} original_slm_inverted.nc
 cdo remapnn,${ancillary_data_path}/grid_0_5.txt original_slm_renamed.nc slm_30min_inverse.nc
 cdo remapnn,${ancillary_data_path}/grid_0_5.txt original_slm_inverted.nc slm_30min.nc
 if $is_coarse_than_half_degree ; then
  cdo remapbil,${ancillary_data_path}/grid_0_5.txt $orography_file_in orography_30min.nc
 else
  cp $orography_file_in orography_30min.nc
 fi

 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh ${no_compile_flag} -r fill_sinks.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r generate_rdirs.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r generate_hdpara_file.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_catchments.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_cumulative_flow.ini -c top_level_driver.cfg

elif $is_finer_than_half_degree ; then

cat > fill_sinks.ini << EOF
[general]
    operation:sink_filling
[sink_filling]
    algorithm:direct_sink_filling
    add_slight_slope_when_filling_sinks:False
    slope_param:0.0
[input_filepaths]
    orography:${orography_file_in}
    landsea:slm.nc
[input_fieldnames]
    orography:${orography_fieldname}
    landsea:slm
[output_filepaths]
    orography_out:filled_orography.nc
[output_fieldnames]
    orography_out:z
EOF

cat > generate_rdirs.ini << EOF
[general]
    operation:determine_river_directions
[river_direction_determination]
    always_flow_to_sea:True
    use_diagonal_nbrs:True
    mark_pits_as_true_sinks:True
[input_filepaths]
    landsea:slm.nc
    orography:filled_orography.nc
[input_fieldnames]
    landsea:slm
    orography:z
[output_filepaths]
    rdirs_out:rdirs.nc
[output_fieldnames]
    rdirs_out:rdirs
EOF

cat > compute_catchments.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:catchment_computation
[input_filepaths]
    rdirs:rdirs.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    loop_logfile:loops.txt
    catchments_out:catch.nc
[output_fieldnames]
    catchments_out:catch
EOF

cat > compute_cumulative_flow.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:cumulative_flow_computation
[input_filepaths]
    rdirs:rdirs.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    cumulative_flow_out:acc.nc
[output_fieldnames]
    cumulative_flow_out:acc
EOF

cat > upscale_rdirs.ini << EOF
[general]
    operation:river_direction_upscaling
    upscaling_factor:${upscaling_factor}
[river_direction_upscaling]
    algorithm:modified_cotat_plus
    parameters_filepath:${cotat_plus_parameters_path}
[input_filepaths]
    fine_rdirs:rdirs.nc
    fine_cumulative_flow:acc.nc
[input_fieldnames]
    fine_rdirs:rdirs
    fine_cumulative_flow:acc
[output_filepaths]
    coarse_rdirs_out:rdirs_pre_loop_removal_30min.nc
[output_fieldnames]
    coarse_rdirs_out:rdirs
EOF

cat > compute_cumulative_flow_pre_loop_removal_30min.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:cumulative_flow_computation
[input_filepaths]
    rdirs:rdirs_pre_loop_removal_30min.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    cumulative_flow_out:acc_pre_loop_removal_30min.nc
[output_fieldnames]
    cumulative_flow_out:acc
EOF

cat > compute_catchments_pre_loop_removal_30min.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:catchment_computation
[input_filepaths]
    rdirs:rdirs_pre_loop_removal_30min.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    loop_logfile:loops_pre_loop_removal_30min.txt
    catchments_out:catch_pre_loop_removal_30min.nc
[output_fieldnames]
    catchments_out:catch
EOF

cat > remove_loops.ini << EOF
[general]
    operation:river_direction_upscaling
    upscaling_factor:${upscaling_factor}
[river_direction_upscaling]
    algorithm:loop_removal
    parameters_filepath:none
[input_filepaths]
    fine_rdirs:rdirs.nc
    fine_cumulative_flow:acc.nc
    coarse_rdirs:rdirs_pre_loop_removal_30min.nc
    coarse_cumulative_flow:acc_pre_loop_removal_30min.nc 
    coarse_catchments:catch_pre_loop_removal_30min.nc
    loop_logfile:loops_pre_loop_removal_30min.txt
[input_fieldnames]
    fine_rdirs:rdirs
    fine_cumulative_flow:acc
    coarse_rdirs:rdirs
    coarse_cumulative_flow:acc
    coarse_catchments:catch
[output_filepaths]
    coarse_rdirs_out:rdirs_30min.nc
[output_fieldnames]
    coarse_rdirs_out:rdirs
EOF

cat > fill_sinks_30min.ini << EOF
[general]
    operation:sink_filling
[sink_filling]
    algorithm:direct_sink_filling
    add_slight_slope_when_filling_sinks:False
    slope_param:0.0
[input_filepaths]
    orography:orography_30min.nc
    landsea:slm_30min.nc
[input_fieldnames]
    orography:${orography_fieldname}
    landsea:slm
[output_filepaths]
    orography_out:filled_orography_30min.nc
[output_fieldnames]
    orography_out:z
EOF

cat > generate_hdpara_file.ini << EOF
[general]
    operation:parameter_generation
[parameter_generation]
    ancillary_data_path:${ancillary_data_path}
    working_dir:${working_directory}/paragen
[input_filepaths]
    landsea:$(pwd)/slm_30min_inverse.nc
    orography:$(pwd)/filled_orography_30min.nc
    rdirs:$(pwd)/rdirs_30min.nc
[input_fieldnames]
    landsea:${landsea_fieldname}
    orography:z
    rdirs:rdirs
[output_filepaths]
    hdpara_out:$(pwd)/final_hdpara.nc
EOF

cat > compute_catchments_30min.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:catchment_computation
[input_filepaths]
    rdirs:rdirs_30min.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    loop_logfile:loops_30min.txt
    catchments_out:catch_30min.nc
[output_fieldnames]
    catchments_out:catch
EOF

cat > compute_cumulative_flow_30min.ini << EOF
[general]
    operation:river_direction_postprocessing
[post_processing]
    algorithm:cumulative_flow_computation
[input_filepaths]
    rdirs:rdirs_30min.nc
[input_fieldnames]
    rdirs:rdirs
[output_filepaths]
    cumulative_flow_out:acc_30min.nc
[output_fieldnames]
    cumulative_flow_out:acc
EOF

 cdo select,name=${landsea_fieldname} ${landsea_file_in} original_slm_selected.nc
 cdo chname,${landsea_fieldname},slm original_slm_selected.nc original_slm_renamed.nc
 cdo expr,"slm=(${landsea_fieldname}==0)" ${landsea_file_in} original_slm_inverted.nc
 cdo remapnn,${ancillary_data_path}/grid_0_5.txt original_slm_renamed.nc slm_30min_inverse.nc
 cdo remapnn,${ancillary_data_path}/grid_0_5.txt original_slm_inverted.nc slm_30min.nc
 cdo remapnn,${orography_file_in} slm_30min.nc slm.nc
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh ${no_compile_flag} -r fill_sinks.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r generate_rdirs.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_cumulative_flow.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_catchments.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r upscale_rdirs.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_catchments_pre_loop_removal_30min.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_cumulative_flow_pre_loop_removal_30min.ini -c top_level_driver.cfg 
 if [[ $(grep -c "[0-9]" "loops_pre_loop_removal_30min.txt") -ne 0 ]]; then
  ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r remove_loops.ini -c top_level_driver.cfg
 else 
  mv rdirs_pre_loop_removal_30min.nc rdirs_30min.nc
 fi
 cdo remapbil,${ancillary_data_path}/grid_0_5.txt $orography_file_in orography_30min.nc
 #Note this second sink filled orography is only used for generating flow parameters and not for river directions
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r fill_sinks_30min.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r generate_hdpara_file.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_cumulative_flow_30min.ini -c top_level_driver.cfg
 ${dynamic_hd_code}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r compute_catchments_30min.ini -c top_level_driver.cfg
fi
