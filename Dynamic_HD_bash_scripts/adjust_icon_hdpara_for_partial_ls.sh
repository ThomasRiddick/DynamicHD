#/bin/bash
set -e

maxland_mask=${1}
minland_mask=${2}
input_hdpara=${3}
output_hdpara=${4}
resolution=${5}

if [[ ${resolution} == 'r2b8' ]]; then
  average_alfk=4.82
elif [[ ${resolution} == 'r2b3' ]]; then
  average_alfk=98.0
elif [[ ${resolution} == 'r2b4' ]]; then
  average_alfk=45.3
elif [[ ${resolution} == 'r2b5' ]]; then
  average_alfk=25.0
elif [[ ${resolution} == 'r2b6' ]]; then
  average_alfk=19.0
elif [[ ${resolution} == 'r2b7' ]]; then
  average_alfk=10.12
elif [[ ${resolution} == 'r2b10' ]]; then
  average_alfk=1.1
else
  echo "Unrecognised resolution"
  exit 1
fi

cdo sub ${maxland_mask} ${minland_mask} difference_in_masks.nc
cdo merge difference_in_masks.nc ${input_hdpara} hdpara_temp.nc
cdo -b F64 expr,"_ls_int=nint(cell_sea_land_mask);ALF_K=(1-_ls_int)*ALF_K+_ls_int*${average_alfk};ALF_N=(1-_ls_int)*ALF_N+_ls_int;AGF_K=(1-_ls_int)*AGF_K+300*_ls_int;FDIR=(1-_ls_int)*FDIR+_ls_int*-1;" hdpara_temp.nc hdpara_temp_altered_int.nc
cdo -b I32 expr,"_ls_int=nint(cell_sea_land_mask);MASK=(1-_ls_int)*MASK;" hdpara_temp.nc hdpara_temp_altered_float.nc
cdo delete,name='FDIR,ALF_K,ALF_N,AGF_K,MASK' ${input_hdpara}  hdpara_temp_unaltered.nc
cdo merge hdpara_temp_altered_int.nc hdpara_temp_altered_float.nc  hdpara_temp_altered.nc
cdo chcode,-1,701,-2,702,-3,703,-4,706,-5,714 hdpara_temp_altered.nc hdpara_temp_altered_fixed_codes.nc
cdo merge  hdpara_temp_unaltered.nc  hdpara_temp_altered_fixed_codes.nc hdpara_orig_order.nc
cdo splitcode hdpara_orig_order.nc hdpara_split 
cdo merge hdpara_split701.nc hdpara_split702.nc hdpara_split703.nc hdpara_split704.nc hdpara_split705.nc hdpara_split706.nc hdpara_split708.nc hdpara_split709.nc hdpara_split714.nc hdpara_split715.nc ${output_hdpara}
rm difference_in_masks.nc
rm hdpara_temp.nc
rm hdpara_temp_altered_int.nc
rm hdpara_temp_altered_float.nc
rm hdpara_temp_altered_fixed_codes.nc
rm hdpara_temp_altered.nc
rm hdpara_temp_unaltered.nc
rm hdpara_orig_order.nc
rm hdpara_split701.nc hdpara_split702.nc hdpara_split703.nc hdpara_split704.nc hdpara_split705.nc hdpara_split706.nc hdpara_split708.nc hdpara_split709.nc hdpara_split714.nc hdpara_split715.nc
