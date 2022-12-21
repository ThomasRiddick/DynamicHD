#!/bin/bash
set -e

echo "Running Compile Script"

compilation_required=${1}
compile_only=${2}
source_directory=${3}
working_directory=${4}
prep_paragen_code=${5}
make_argument=${6:-"compile_only"}

#Compile C++ and Fortran Code if this is the first timestep
if $compilation_required ; then
  cd ${source_directory}
  rm makefile
  ./config/dkrz/levante.gcc-11.2.0
  make -f makefile clean
  make -f makefile ${make_argument}
  cd - 2>&1 > /dev/null
fi

if ${prep_paragen_code} && ! ${compile_only} ; then
  # Clean up paragen working directory if any
  if [[ -d "${working_directory}/paragen" ]]; then
    cd ${working_directory}/paragen
    rm -f paragen.inp soil_partab.txt slope.dat riv_vel.dat riv_n.dat riv_k.dat over_vel.dat over_n.dat over_k.dat || true
    rm -f hdpara.srv global.inp ddir.inp bas_k.dat || true
    cd - 2>&1 > /dev/null
    rmdir ${working_directory}/paragen
  fi
fi

#Prepare bin directory for python code and bash code
if $compilation_required; then
  mkdir -p ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/bin
  mkdir -p ${source_directory}/Dynamic_HD_bash_scripts/bin
fi

#Compile fortran code used called shell script wrappers
if $compilation_required && $prep_paragen_code; then
  if [[ -e "${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/.git" ]]; then
    echo "Compiling Fortran code called from shell script wrappers"
    ${source_directory}/Dynamic_HD_bash_scripts/compile_paragen_and_hdfile.sh ${source_directory}/Dynamic_HD_bash_scripts/bin ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/fortran ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/fortran/paragen.f paragen
  else
    echo "No parameter_generation_scripts submodule; this is an error if parameter generation is required but otherwise is not"
  fi
fi

