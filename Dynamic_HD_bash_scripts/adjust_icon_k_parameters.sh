#!/bin/bash

icon_para_input_filename=${1}
icon_para_output_filename=${2}
resolution=${3}

if [[ ${resolution} == 'r2b9' ]]; then
  echo "Retuning included in paragen program for r2b9... "
  cp ${icon_para_input_filename} ${icon_para_output_filename}
  exit 0
elif [[ ${resolution} == 'r2b8' ]] || [[ ${resolution} == 'r3b7' ]]; then
  new_arf_k=0.027
  arf_k_adjustment_thres=0.1
  new_alf_k=4.82
  alf_k_adjustment_thres=8.0
elif [[ ${resolution} == 'r2b3' ]]; then
  new_arf_k=0.54
  arf_k_adjustment_thres=2.5
  new_alf_k=98.0
  alf_k_adjustment_thres=400.0
elif [[ ${resolution} == 'r2b4' ]]; then
  new_arf_k=0.25
  arf_k_adjustment_thres=1.0
  new_alf_k=45.3
  alf_k_adjustment_thres=200.0
elif [[ ${resolution} == 'r2b5' ]]; then
  new_arf_k=0.135
  arf_k_adjustment_thres=0.75
  new_alf_k=25.0
  alf_k_adjustment_thres=150.0
elif [[ ${resolution} == 'r2b6' ]]; then
  new_arf_k=0.105
  arf_k_adjustment_thres=0.4
  new_alf_k=19.0
  alf_k_adjustment_thres=50.0
elif [[ ${resolution} == 'r2b7' ]]; then
  new_arf_k=0.056
  arf_k_adjustment_thres=0.2
  new_alf_k=10.12
  alf_k_adjustment_thres=40.0
elif [[ ${resolution} == 'r2b10' ]]; then
  new_arf_k=0.006
  arf_k_adjustment_thres=0.025
  new_alf_k=1.1
  alf_k_adjustment_thres=4.0
elif [[ ${resolution} == 'r2b11' ]]; then
  new_arf_k=0.0032
  arf_k_adjustment_thres=0.0058
  new_alf_k=0.57
  alf_k_adjustment_thres=1.06
else
  echo "Unrecognised resolution"
  exit 1
fi

ncap2 -s "ALF_K=(float(ALF_K<=${alf_k_adjustment_thres})*ALF_K)+(float(ALF_K>${alf_k_adjustment_thres})*${new_alf_k});ARF_K=(float(ARF_K<=${arf_k_adjustment_thres})*ARF_K)+(float(ARF_K>${arf_k_adjustment_thres})*${new_arf_k});" ${icon_para_input_filename} ${icon_para_output_filename}_temp
cdo replace ${icon_para_input_filename} ${icon_para_output_filename}_temp ${icon_para_output_filename}
rm ${icon_para_output_filename}_temp
