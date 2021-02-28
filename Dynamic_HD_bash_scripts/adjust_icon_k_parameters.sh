#!/bin/bash

icon_para_input_filename=${1}
icon_para_output_filename=${2}
resolution=${3}

if [[ ${resolution} == 'r2b8' ]]; then
  new_arf_k=0.027
  arf_k_adjustment_thres=0.1
  new_alf_k=4.82
  alf_k_adjustment_thres=8.0
elif [[ ${resolution} == 'r2b4' ]]; then
  new_arf_k=0.25
  arf_k_adjustment_thres=1.0
  new_alf_k=45.3
  alf_k_adjustment_thres=200.0
else
  exit 1
fi

ncap2 -s "ALF_K=(float(ALF_K<=${alf_k_adjustment_thres})*ALF_K)+(float(ALF_K>${alf_k_adjustment_thres})*${new_alf_k});ARF_K=(float(ARF_K<=${arf_k_adjustment_thres})*ARF_K)+(float(ARF_K>${arf_k_adjustment_thres})*${new_arf_k});" ${icon_para_input_filename} ${icon_para_output_filename}_temp
cdo replace ${icon_para_input_filename} ${icon_para_output_filename}_temp ${icon_para_output_filename}
rm ${icon_para_output_filename}_temp
