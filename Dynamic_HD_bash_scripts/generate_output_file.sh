#!/bin/bash

#Set standard scripting options
set -e

#Get command line arguments
bin_dir=${1}
src_dir=${2}
rdir_file=${3}
ls_mask_file=${4}
null_file=${5}
area_spacing_file=${6}
hd_grid_specs_file=${7}
output_file=${8}
work_dir=${9}

#Set compiler
comp=gfortran

#Set CDO
if [[ $OSTYPE = "darwin"* ]]; then
	cdo=/usr/local/bin/cdo
else
	cdo=cdo	
fi

# **** Zusammenfassen der Parameterdateien in Inputdatei hdpara.srv
# **** Combining the parameter files in Input File hdpara.srv  
echo 'Step 8: hdpara.srv erzeugen'
echo ${work_dir} > ${work_dir}/ddir.inp
cat > ${work_dir}/global.inp << EOF1
Initialisierungsdatei fuer Programm GLOBSIM, d.h. fuer Routine GLOBINP auf LAKE:

IQUE  : Kommentarvariable ( 0 = Kein Kommentar )
0
IOUT  : Mittelungsartvariable fuer Output: 1=30d, 2=10d, 3=7d,4=Monthly 
1
IBASE : Baseflow AN (1) oder AUS (0)
1
ISREAD: Initialisierung der Speicherfelder: 0 = AUS, sonst AN
1
ISWRIT: Zwischenspeicherung der Speicherfelder: 0 = AUS, sonst Speicherintervall
359
NSTEP : Anzahl der zu verarbeitenden Timesteps vom Anfang bis NSTEP
359
JAHR1 : Startjahr der Messreihen
1
ISOLOG: Logausgabe Isodatei (0=Keine, 1=Bothnian Bay/Sea, 2=Torneaelven, 3=glob)
1
IINDEX: Add RDF-Index fields to HD parameter file: 1/0 = ON/OFF
1
TDNMAS: Dateiname der Landmaske
${ls_mask_file}
TDNRES: Dateiname des binaeren globalen Reservoir-Speicherarrays: reservoir.dat
${null_file}
TDNDIR: Dateiname des Riverdirection-Files
${rdir_file}
TDNLFK: Dateiname fuer Overlandflow k-Parameter
over_k.dat
TDNLFN: Dateiname fuer Overlandflow n-Parameter
over_n.dat
TDNRFK: Dateiname fuer Riverflow k-Parameter
riv_k.dat
TDNRFN: Dateiname fuer Riverflow n-Parameter
riv_n.dat
TDNGFK: Dateiname fuer Baseflow k-Parameter
bas_k.dat
TDNGSP: Dateiname fuer linearen Baseflow-Speicher, Initialisierungszustand
${null_file}
TDNINF: Dateiname fuer die Inflow-Initialisierung
${null_file}
TDNARE: Dateiname fuer globalen Laengen- und Flaechenfile
${area_spacing_file}
The End
EOF1
#
cd ${work_dir}
${comp} -o ${bin_dir}/hdfile ${src_dir}/hdfile.f ${src_dir}/globuse.f
${bin_dir}/hdfile
cd -
#
# *************  Conversion to NetCDF
echo 'Conversion to NetCDF'
cat > ${work_dir}/soil_partab.txt << EOF2
&parameter
 param        = 172
 out_name     = FLAG 
 long_name    = "HD model land sea mask"
 /
&parameter
 param        = 701
 out_name     = FDIR 
 long_name    = "River Direction File = RDF"
 /
&parameter
 param        = 702
 out_name     = ALF_K
 long_name    = "HD model parameter Overland flow k"
 /
&parameter
 param        = 703
 out_name     = ALF_N
 long_name    = "HD model parameter Overland flow n"
 /
&parameter
 param        = 704
 out_name     = ARF_K
 long_name    = "HD model parameter Riverflow k"
 /
&parameter
 param        = 705
 out_name     = ARF_N
 long_name    = "HD model parameter Riverflow n"
 /
&parameter
 param        = 706
 out_name     = AGF_K
 long_name    = "HD model parameter Baseflow k"
 /
&parameter
 param        = 707
 out_name     = AREA
 long_name    = "Areas at 5 Min grid [m2] = f(latitude)"
 /
&parameter
 param        = 708
 out_name     = FILNEW
 long_name    = "Longitude index of Flow Destination according to FDIR"
 /
&parameter
 param        = 709
 out_name     = FIBNEW
 long_name    = "Latitude index of Flow Destination according to FDIR"
 /
EOF2
#
CGRID=${hd_grid_specs_file}
cd ${work_dir} 
${cdo} -f nc setpartabp,soil_partab.txt -setgrid,${CGRID} hdpara.srv ${output_file}
cd - 1>&2 > /dev/null
