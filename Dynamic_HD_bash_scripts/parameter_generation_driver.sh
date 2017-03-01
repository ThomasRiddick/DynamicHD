#!/bin/sh

#Set standard scripting options
set -e

#Get command line arguments
bin_dir=${1}
src_dir=${2}
rdir_file=${3}
topography_file=${4}
inner_slope_file=${5}
ls_mask_file=${6}
null_file=${7}
area_spacing_file=${8}
orography_variance_file=${9}
paragen_source_filepath=${10}
paragen_bin_file=${11}
work_dir=${12}

#Set compiler
comp=/usr/local/bin/gfortran

#Setup working directory
mkdir ${work_dir}

#Setup input file

cat > ${work_dir}/paragen.inp << EOF
Initialization data for the program PARAGEN:

IPARA : Type of Parameterization (1 = Sausen-Analogie)
8
IFIN  : Input format (1 = Cray binary format, 2 = REGEN, Glob. binary format, )
2
IFOUT : Output format (3 = Waveiso-Format)
2
IQUE  : Comments ( 0 = No comments )
0
IGMEM : Base storage flow initialization No/Yes (1/0)
0
IBASE : Base flow initialization type (0:k=300 days, 1: 0 with dx-Abh., 2,3=Beate)
3
TDNFL : Filename of the global Area/Spacing array
${area_spacing_file}
TDNORO: Filename of the global Orography Arrays
${topography_file}
TDNSIG: Filename of the global Orography variance array
${orography_variance_file}
TDNMAS: Filename of the landsea mask
${ls_mask_file}
TDNGMA: Filename of the glacier mask, z.B. null.dat
${null_file}
TDNDIR: Filename of the  River Direction File
${rdir_file}
TDNDD : Filename of the Drainage Density Arrays
${null_file}
TDNSLI: Filename of the Inner Slope-File
${inner_slope_file}
TDNLAK: Filename of the Lake-Percentage-File
${null_file}
TDNSWA: Filename of the Swamp-Percentage-File
${null_file}
TDNMWT: Filename of the Matthews Wetland Type
${null_file}
TDNPER: Filename of the Permafrost-File
${null_file}
ILAMOD: Model of Lake-Dependence (0=without Lakes,1=Charbonneau, 2=tanh)
0
ISWMOD: Model of Swamp-Dependence (0=without Swamps, 1=swamps+lakes, 2=tanh, 4=Ov.)
0
VLA100: Flow-Velocity for 100 % Lake-Percentage [m/s] (0.0003 by tanh)
0.01
VSW100: Flow-Velocity for 100 % Swamp-Percentage [m/s]: 0.077 = 200 km/month
0.06
PROARE: Area Percentage, from which the Lake- or Swamp-Percentage takes effect
50.
FK_LFK: Modification factor for k-value for Overlandflow
1.
FK_LFN: Modification factor for n-value for Overlandflow
1.
FK_RFK: Modification factor for k-value for Riverflow
1.
FK_RFN: Modification factor for n-value for Riverflow
1.
FK_GFK: Modification factor for k-value for Baseflow
1.
The End
EOF

cd ${work_dir}

${comp} -o ${bin_dir}/${paragen_bin_file} ${paragen_source_filepath} ${src_dir}/globuse.f ${src_dir}/mathe.f ${src_dir}/modtime.f
${bin_dir}/${paragen_bin_file}

cd -