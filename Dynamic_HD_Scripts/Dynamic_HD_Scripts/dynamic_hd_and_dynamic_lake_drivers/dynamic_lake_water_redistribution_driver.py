import argparse
import cdo
import os.path as path
from Dynamic_HD_Scripts.tools.dynamic_lake_operators \
    import advanced_water_redistribution_driver

def redistribute_water(input_lake_numbers_filepath,
                       input_lake_numbers_fieldname,
                       input_lake_centers_filepath,
                       input_lake_centers_fieldname,
                       input_water_to_redistribute_filepath,
                       input_water_to_redistribute_fieldname,
                       output_lakestart_filepath,
                       working_directory):
    output_water_redistributed_to_lakes_file=\
        path.join(working_directory,"water_to_lakes_temp.nc")
    output_water_redistributed_to_rivers_file=\
        path.join(working_directory,"water_to_rivers_temp.nc")
    advanced_water_redistribution_driver(input_lake_numbers_file=
                                         input_lake_numbers_filepath,
                                         input_lake_numbers_fieldname=
                                         input_lake_numbers_fieldname,
                                         input_lake_centers_file=
                                         input_lake_centers_filepath,
                                         input_lake_centers_fieldname=
                                         input_lake_centers_fieldname,
                                         input_water_to_redistribute_file=
                                         input_water_to_redistribute_filepath,
                                         input_water_to_redistribute_fieldname=
                                         input_water_to_redistribute_fieldname,
                                         output_water_redistributed_to_lakes_file=
                                         output_water_redistributed_to_lakes_file,
                                         output_water_redistributed_to_lakes_fieldname=
                                         "water_redistributed_to_lakes",
                                         output_water_redistributed_to_rivers_file=
                                         output_water_redistributed_to_rivers_file,
                                         output_water_redistributed_to_rivers_fieldname="water_redistributed_to_rivers",
                                         coarse_grid_type="HD")
    cdo_inst = cdo.Cdo()
    cdo_inst.merge(input=" ".join([output_water_redistributed_to_lakes_file,
                                   output_water_redistributed_to_rivers_file]),
                   output=output_lakestart_filepath)

class Arguments:
    """An empty class used to pass namelist arguments into the main routine as keyword arguments."""

    pass

def parse_arguments():
    """Parse the command line arguments using the argparse module.

    Returns:
    An Arguments object containing the comannd line arguments.
    """

    args = Arguments()
    parser = argparse.ArgumentParser("Update river flow directions")
    parser.add_argument('input_lake_numbers_filepath',
                        metavar='input-lake-numbers-filepath',
                        help='Full path to lake numbers file to use',
                        type=str)
    parser.add_argument('input_lake_centers_filepath',
                        metavar='input-lake-centers-filepath',
                        help='Full path to lake centers to use',
                        type=str)
    parser.add_argument('input_water_to_redistribute_filepath',
                        metavar='input-water-to-redistribute-filepath',
                        help='Full path to file containing water from lakes in previous run',
                        type=str)
    parser.add_argument('output_lakestart_filepath',
                        metavar='output-lakestart-filepath',
                        help='Full path to target destination for output lakestart file ',
                        type=str)
    parser.add_argument('working_directory',
                        metavar='working-directory',
                        help='Path to working directory to use',
                        type=str)
    parser.add_argument('input_lake_numbers_fieldname',
                        metavar='input-lake-numbers-filepath',
                        help='Fieldname of lake numbers field',
                        type=str)
    parser.add_argument('input_lake_centers_fieldname',
                        metavar='input-lake-centers-filepath',
                        help='Fieldname of lake centers field',
                        type=str)
    parser.add_argument('input_water_to_redistribute_fieldname',
                        metavar='input-water-to-redistribute-fieldname',
                        help='Fieldname of water to input water to redistribute',
                        type=str)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    redistribute_water(**vars(args))
