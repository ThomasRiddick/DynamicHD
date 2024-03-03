# DynamicHD
Tools for Dynamic River Routing for Paleoclimate Simulations
Code to accompany the paper "Dynamic hydrological discharge modelling for coupled climate model simulations of the last glacial cycle" (Riddick et al 2018, Submitted to GMD). 
This paper gives references for the various algorithms used and describes those that were developed specifically for this project.
This code was written as part of the BMBF funded PalMod Project at the Max Planck Institute for Meteorology by Thomas Riddick.
The top level script is Dynamic_HD_Code/Dynamic_HD_bash_scripts/dynamic_hd_top_level_driver.sh
Individual tools can be run using Dynamic_HD_Code/Dynamic_HD_bash_scriptshd_operator_driver.sh but this is currently untested.
A small sub-section of this code is not present in this repository for licensing reasons. This would go in the directory /Dynamic_HD_Code/Dynamic_HD_bash_scripts/parameter_generation_scripts
and anyone wishing to run the top level script would need to develop a replacement for it compatilble with there own Hydrological Discharge Model. Many of the individual tools can be written without this
The conda environment used to run this can be created by installing conda from https://docs.anaconda.com/free/miniconda/ then running the follow at the command line (linux/os x):
conda config --add channels conda-forge
conda env create -f dyhdenv3.yaml  
This project imports the python library pysimplegui 4.60.5 which is avaiable here: https://anaconda.org/conda-forge/PySimpleGUI
