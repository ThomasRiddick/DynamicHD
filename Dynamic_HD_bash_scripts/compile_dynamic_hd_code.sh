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
  echo "C++ source directory"
  echo ${source_directory}/Dynamic_HD_Cpp_Code
  echo "Compiling C++ code" 1>&2
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src/base
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src/algorithms
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src/drivers
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src/testing
  mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src/command_line_drivers
  cd ${source_directory}/Dynamic_HD_Cpp_Code/Release
  make -f ../makefile clean
  make -f ../makefile ${make_argument}
  cd - 2>&1 > /dev/null
  echo "Fortran source directory"
  echo ${source_directory}/Dynamic_HD_Fortran_Code
  echo "Compiling Fortran code" 1>&2
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src/algorithms
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src/base
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src/command_line_drivers
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src/drivers
  mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src/testing
  cd ${source_directory}/Dynamic_HD_Fortran_Code/Release
  make -f ../makefile clean
  make -f ../makefile ${make_argument}
  cd - 2>&1 > /dev/null
fi

# Clean shared libraries
if $compilation_required; then
  cd ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts
    make -f makefile clean
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

#Setup cython interface between python and C++
if $compilation_required; then
  cd ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/interface/cpp_interface
  echo "Compiling Cython Modules" 1>&2
  python ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/interface/cpp_interface/setup_fill_sinks.py clean --all
  python ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/interface/cpp_interface/setup_fill_sinks.py build_ext --inplace -f
  cd - 2>&1 > /dev/null
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

