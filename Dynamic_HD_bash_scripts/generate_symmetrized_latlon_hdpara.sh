#!/bin/bash

global_north=${1}
ls_mask=${2}
symmetrized_hdpara_ancillary_data=${3}
workdir=${4}

# Define config file
config_filepath="${symmetrized_hdpara_ancillary_data}/top_level_driver.cfg"

# Check config file exists  and has correct format

if ! [[ -f ${config_filepath} ]]; then
  echo "Top level script config file (${config_filepath}) doesn't exist!"
  exit 1
fi

if egrep -v -q "^(#.*|.*=.*)$" ${config_filepath}; then
  echo "Config file has wrong format" 1>&2
  exit 1
fi

# Read in source_directory
source ${config_filepath}

# Check we have actually read the variables correctly
if [[ -z ${source_directory} ]]; then
  echo "Source directory not set in config file or set to a blank string" 1>&2
  exit 1
fi

original_ref_orography_filename=ice5g_v1_2_00_0k_10min.nc
original_orography_corrections_filename=orog_corrs_field_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_20170517_003802_with_grid.nc
original_ref_orography=${symmetrized_hdpara_ancillary_data}/${original_ref_orography_filename}
original_orography_corrections=${symmetrized_hdpara_ancillary_data}/${original_orography_corrections_filename}
symmetrized_ref_orography=${workdir}/ref_orography_sym.nc
symmetrized_orography_corrections=${workdir}/orography_corrections_sym.nc
symmetrized_ref_orography_reflection=${workdir}/ref_orography_symmetrized_refl.nc
symmetrized_orography_corrections_reflection=${workdir}/orography_corrections_sym_refl.nc
symmetrized_hdpara_northern_hemisphere=${workdir}/hdpara_north.nc
symmetrized_hdpara_southern_hemisphere=${workdir}/hdpara_south.nc
symmetrized_hdpara=${workdir}/hdpara_symmetrized_for_global_north_or_south.nc
symmetrized_hdpara_gen_ancillary_directory_north=${workdir}/HD_ancillary_north
symmetrized_hdpara_gen_ancillary_directory_south=${workdir}/HD_ancillary_south

cp -r ${symmetrized_hdpara_ancillary_data} \
  ${symmetrized_hdpara_gen_ancillary_directory_north}

cd ${symmetrized_hdpara_gen_ancillary_directory_north}
if ${global_north}; then
  ln -s ${symmetrized_ref_orography} \
    ${original_ref_orography_filename}
  ln -s ${symmetrized_orography_corrections} \
    ${original_orography_corrections_filename}
else
  ln -s ${symmetrized_ref_orography_reflection} \
    ${original_ref_orography_filename}
  ln -s ${symmetrized_orography_corrections_reflection} \
    ${original_orography_corrections_filename}
fi

cp -r ${symmetrized_hdpara_ancillary_data} \
  ${symmetrized_hdpara_gen_ancillary_directory_south}

cd ${symmetrized_hdpara_gen_ancillary_directory_south}
if ${global_north}; then
  ln -s ${symmetrized_ref_orography_reflection} \
    ${original_ref_orography_filename}
  ln -s ${symmetrized_orography_corrections_reflection} \
    ${original_orography_corrections_filename}
else
  ln -s ${symmetrized_ref_orography} \
    ${original_ref_orography_filename}
  ln -s ${symmetrized_orography_corrections} \
    ${original_orography_corrections_filename}
fi

#Create orography and correct set
if ${global_north}; then
  cdo -setcindexbox,8000.0,1,2048,513,513 -setcindexbox,-8000.0,1,2048,514,1024 \
    ${original_ref_orography} ${symmetrized_ref_orography}
  cdo symmetrize ${symmetrized_ref_orography} ${symmetrized_ref_orography_reflection}
  cdo -setclatlonbox,0.0,-180,180,0,-90 \
    ${original_orography_corrections} ${symmetrized_orography_corrections}
  cdo symmetrize ${symmetrized_orography_corrections} ${symmetrized_orography_corrections_reflection}
else
  cdo -setcindexbox,8000.0,1,2048,512,512 -setcindexbox,-8000.0,1,2048,0,511 \
    ${original_ref_orography} ${symmetrized_ref_orography}
  cdo symmetrize ${symmetrized_ref_orography} ${symmetrized_ref_orography_reflection}
  cdo -setclonlatbox,0.0,-180,180,90,0 \
    ${original_orography_corrections} ${symmetrized_orography_corrections}
  cdo symmetrize ${symmetrized_orography_corrections} ${symmetrized_orography_corrections_reflection}
fi

#Create null input file
cdo const,0.0,${original_ref_orography} ${workdir}/null.nc

#Run dynamic hd scripting for first hemisphere
first_timestep=False
input_orography_filepath=${workdir}/null.nc
input_ls_mask_filepath=${ls_mask}
present_day_base_orography_filepath=${workdir}/null.nc
glacier_mask_filepath=${workdir}/null.nc
output_hdpara_filepath=${symmetrized_hdpara_northern_hemisphere}
ancillary_data_directory=${symmetrized_hdpara_gen_ancillary_directory_north}
working_directory=${workdir}
diagnostic_output_directory=${workdir}/diag_north
diagnostic_output_exp_id_label=none
diagnostic_output_time_label=0
output_hdstart_filepath=${workdir}/null_out_north.nc
${source_directory}/Dynamic_HD_bash_scripts/dynamic_hd_top_level_driver.sh \
  ${first_timestep} \
  ${input_orography_filepath} \
  ${input_ls_mask_filepath} \
  ${present_day_base_orography_filepath} \
  ${glacier_mask_filepath} \
  ${output_hdpara_filepath} \
  ${ancillary_data_directory} \
  ${working_directory} \
  ${diagnostic_output_directory} \
  ${diagnostic_output_exp_id_label} \
  ${diagnostic_output_time_label} \
  ${output_hdstart_filepath}

#Run dynamic hd scripting for second hemisphere
  first_timestep=False
  input_orography_filepath=${workdir}/null.nc
  input_ls_mask_filepath=${ls_mask}
  present_day_base_orography_filepath=${workdir}/null.nc
  glacier_mask_filepath=${workdir}/null.nc
  output_hdpara_filepath=${symmetrized_hdpara_southern_hemisphere}
  ancillary_data_directory=${symmetrized_hdpara_gen_ancillary_directory_south}
  working_directory=${workdir}
  diagnostic_output_directory=${workdir}/diag_south
  diagnostic_output_exp_id_label=none
  diagnostic_output_time_label=0
  output_hdstart_filepath=${workdir}/null_out_south.nc
${source_directory}/Dynamic_HD_bash_scripts/dynamic_hd_top_level_driver.sh \
  ${first_timestep} \
  ${input_orography_filepath} \
  ${input_ls_mask_filepath} \
  ${present_day_base_orography_filepath} \
  ${glacier_mask_filepath} \
  ${output_hdpara_filepath} \
  ${ancillary_data_directory} \
  ${working_directory} \
  ${diagnostic_output_directory} \
  ${diagnostic_output_exp_id_label} \
  ${diagnostic_output_time_label} \
  ${output_hdstart_filepath}

#Combine results
cdo setclonlatbox,1,-180,180,0,-90 - const,0,${original_ref_orography} ${workdir}/is_south.nc
cdo cond2 ${workdir}/is_south.nc ${symmetrized_hdpara_southern_hemisphere} ${symmetrized_hdpara_northern_hemisphere} ${workdir}/symmetrized_hdpara_temp.nc

#Redo outflow points
cat > mark_rivermouths.ini << EOF
[general]
    operation:mark_river_mouths
[input_filepaths]
    landsea:${ls_mask}
    rdirs:${workdir}/symmetrized_hdpara_temp.nc
[input_fieldnames]
    landsea:slm
    rdirs:FDIR
[output_filepaths]
    rdirs_out:${workdir}/rdirs_out.nc
[output_fieldnames]
    rdirs_out:FDIR
EOF

${source_directory}/Dynamic_HD_bash_scripts/hd_operator_driver.sh -n -r \
    mark_rivermouths.ini \
    -c ${symmetrized_hdpara_ancillary_data}/top_level_driver.cfg
cdo replace ${workdir}/symmetrized_hdpara_temp.nc \
    ${workdir}/rdirs_out.nc ${symmetrized_hdpara}
