#!/bin/bash
set -e

echo "Running Compile Script"

compilation_required=${1}
compile_only=${2}
source_directory=${3}
working_directory=${4}
prep_paragen_code=${5}

#Compile C++ and Fortran Code if this is the first timestep
if $compilation_required ; then
  cd ${source_directory}
  make clean
  make
  cd - 2>&1 > /dev/null
fi

if ${prep_paragen_code} && ! ${compile_only} ; then
  # Clean up paragen working directory if any
  rm -rf ${working_directory}/paragen
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

