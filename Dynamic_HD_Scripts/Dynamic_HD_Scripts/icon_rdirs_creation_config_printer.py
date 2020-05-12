'''
Print information from the config file for coarse icon river direction generation for use by bash
Created on May 9, 2020

@author: thomasriddick
'''

import create_icon_coarse_river_directions_driver as cicrdd
import argparse

def print_filenames_from_config(python_config_filename_in):
    driver_object = cicrdd.Icon_Coarse_River_Directions_Creation_Drivers()
    driver_object.python_config_filename = python_config_filename_in
    config = driver_object._read_and_validate_config()
    print (config.get("input_options","ten_minute_corrected_orography_filename")
           + " " +
           config.get("input_fieldname_options",
                      "ten_minute_corrected_orography_fieldname"))

class Arguments(object):
    """An empty class used to pass namelist arguments into the main routine as keyword arguments."""

    pass

def parse_arguments():
    """Parse the command line arguments using the argparse module.

    Returns:
    An Arguments object containing the comannd line arguments.
    """

    args = Arguments()
    parser = argparse.ArgumentParser("Update river flow directions")
    parser.add_argument('python_config_filename',
                        metavar='python-config-filename',
                        help='Full path to python configuration file',
                        type=str)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    print_filenames_from_config(args.python_config_filename)
