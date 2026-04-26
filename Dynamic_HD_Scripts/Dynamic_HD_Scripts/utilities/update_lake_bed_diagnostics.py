import xarray as xr
import numpy as np
import argparse

null_value = -9999

def update_lake_bed_diagnostics(lake_mask,is_lake,was_lake,interval):
    is_lake[lake_mask & (is_lake >= 0)] += interval
    is_lake[lake_mask & (is_lake < 0)]  = 0
    was_lake[lake_mask] = null_value
    was_lake[np.logical_not(lake_mask) & (was_lake > 0)] += interval
    was_lake[np.logical_not(lake_mask) & (was_lake <= 0)] = null_value
    was_lake[np.logical_not(lake_mask) & (is_lake >= 0)] = interval
    is_lake[np.logical_not(lake_mask)] = null_value

def update_lake_bed_diagnostics_driver(lake_mask_filepath,
                                       input_lake_bed_diagnostics_filepath,
                                       output_lake_bed_diagnostics_filepath,
                                       lake_mask_fieldname,
                                       interval):
    lake_mask = xr.open_dataset(lake_mask_filepath)[lake_mask_fieldname].data
    lake_bed_diagnostics_ds = xr.open_dataset(input_lake_bed_diagnostics_filepath)
    is_lake = lake_bed_diagnostics_ds["is_lake"].data
    was_lake = lake_bed_diagnostics_ds["was_lake"].data
    update_lake_bed_diagnostics(lake_mask,is_lake,was_lake,interval)
    lake_bed_diagnostics_ds.to_netcdf(output_lake_bed_diagnostics_filepath)

class Arguments:
    """An empty class used to pass namelist arguments into the main routine as keyword arguments."""

    pass

def parse_arguments():
    """Parse the command line arguments using the argparse module.

    Returns:
    An Arguments object containing the command line arguments.
    """

    args = Arguments()
    parser = argparse.ArgumentParser("Update lake bed diagnostics")
    parser.add_argument('lake_mask_filepath',
                        metavar='lake-mask-filepath',
                        help='Full path to file containing current lake mask',
                        type=str)
    parser.add_argument('input_lake_bed_diagnostics_filepath',
                        metavar='input-lake-bed-diagnostics-filepath',
                        help='Input lake bed diagnostics filepath',
                        type=str)
    parser.add_argument('output_lake_bed_diagnostics_filepath',
                        metavar='output-lake-bed-diagnostics-filepath',
                        help='Output lake bed diagnostics filepath',
                        type=str)
    parser.add_argument('lake_mask_fieldname',
                        metavar='lake-mask-fieldname',
                        help='Current lake mask fieldname',
                        type=str)
    parser.add_argument('interval',
                        help='Interval between run sections',
                        type=int)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    update_lake_bed_diagnostics_driver(**vars(args))
