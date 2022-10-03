#/bin/bash
set -e

#This script move the content of FRFMEM1-5 and FLFMEM and FGMEM to FINFL where the river directions
#show either an outlet, sea or a sink point. This will enable the HD model to release this water
#into the sea

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
        module load ${module_name} 1.9.2-magicsxx-intel1
else
	export MODULEPATH="/sw/common/Modules:/client/Modules"
        eval "eval `/usr/bin/tclsh /sw/share/Modules/modulecmd.tcl bash load ${module_name}`"
fi
}

function find_abs_path
{
relative_path=$1
perl -MCwd -e 'print Cwd::abs_path($ARGV[0]),qq<\n>' $relative_path
}

#Process command line arguments
input_hdpara_filepath=${1}
input_hdstart_filepath=${2}
output_hdstart_filepath=${3}
working_dir=${4}
no_modules=${5:-false}

#Load necessary modules
if ! ${no_modules} && echo $LOADEDMODULES | fgrep -q -v "cdo" ; then
  load_module cdo
  load_module nco
fi

#convert to absolute path names
input_hdpara_filepath=$(find_abs_path $input_hdpara_filepath)
input_hdstart_filepath=$(find_abs_path $input_hdstart_filepath)
output_hdpara_filepath=$(find_abs_path $output_hdpara_filepath)

#Change to the specified working directory
cd ${working_dir}
#Extract the river directions from the hdpara file and copy them to a temporary version of the
#hdstart file
cdo select,name=FDIR ${input_hdpara_filepath} rdirs.nc
cdo merge rdirs.nc ${input_hdstart_filepath} hdstart_temp.nc
#Prepare a set of instructions for transferring water to the FINFL field from other
#reserviors
cat >move_res_to_rflow_inst.txt << EOL
_mask=(FDIR==0.0||FDIR==-1.0||FDIR==5.0);
FINFL=_mask ? FINFL+FGMEM : FINFL;
FGMEM=_mask ? 0 : FGMEM;
FINFL=_mask ? FINFL+FLFMEM : FINFL;
FLFMEM=_mask ? 0 : FLFMEM;
FINFL=_mask ? FINFL+FRFMEM1 : FINFL;
FRFMEM1=_mask ? 0 : FRFMEM1;
FINFL=_mask ? FINFL+FRFMEM2 : FINFL;
FRFMEM2=_mask ? 0 : FRFMEM2;
FINFL=_mask ? FINFL+FRFMEM3 : FINFL;
FRFMEM3=_mask ? 0 : FRFMEM3;
FINFL=_mask ? FINFL+FRFMEM4 : FINFL;
FRFMEM4=_mask ? 0 : FRFMEM4;
FINFL=_mask ? FINFL+FRFMEM5 : FINFL;
FRFMEM5=_mask ? 0 : FRFMEM5;
EOL
#Apply instructions
cdo aexprf,'move_res_to_rflow_inst.txt' hdstart_temp.nc hdstart_temp2.nc
#Clean up hdstart file, add important global attributes and give it its final filename
cdo delete,name=FDIR hdstart_temp2.nc hdstart_temp3.nc
ncatted -a istep,global,c,l,1 hdstart_temp3.nc hdstart_temp4.nc
ncatted -a hd_steps_per_day,global,c,l,1 hdstart_temp4.nc hdstart_temp5.nc
ncatted -a riverflow_timestep,global,c,l,4 hdstart_temp5.nc ${output_hdstart_filepath}
#Delete temporary files and change back to original directory
rm hdstart_temp.nc hdstart_temp2.nc hdstart_temp3.nc hdstart_temp4.nc
rm hdstart_temp5.nc move_res_to_rflow_inst.txt rdirs.nc
cd -
