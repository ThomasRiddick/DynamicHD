'''
Merge Hydrosheds river directions for South America, Africa and Australia and
the corrected set of river directions for North America, Central America, Asia and
Europe then remove some/all endorheic basin (as specified) and correct any loops

Created on March 3, 2020

@author: thomasriddick
'''

import dynamic_hd_driver as dyn_hd_dr
import compute_catchments as comp_catchs
import field
import dynamic_hd
import iodriver
import utilities
import tempfile
import argparse
import numpy as np
import os
import os.path as path
import warnings
import ConfigParser
from flow_to_grid_cell import create_hypothetical_river_paths_map
import libs.fill_sinks_wrapper as fill_sinks_wrapper

class Icon_Coarse_River_Directions_Creation_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):
    """Drivers for running a productions run of the ICON coarse river direction creation code"""

    def __init__(self,ls_mask_filename=None,output_rdirs_filename=None,output_catchments_filename=None,
                 output_cumulative_flow_filename=None,true_sinks_filename=None,keep_all_internal_basins=False,
                 python_config_filename=None,working_directory=None,rerun_post_processing_only=False):
        self.ls_mask_filename = ls_mask_filename
        self.output_rdirs_filename = output_rdirs_filename
        self.output_catchments_filename = output_catchments_filename
        self.output_cumulative_flow_filename = output_cumulative_flow_filename
        self.true_sinks_filename = true_sinks_filename
        self.keep_all_internal_basins = keep_all_internal_basins
        self.python_config_filename = python_config_filename
        self.working_directory = working_directory
        self.rerun_post_processing_only = rerun_post_processing_only

    def trial_generate_r2b4_mask_10min_combined_river_directions(self):
        super(Icon_Coarse_River_Directions_Creation_Drivers,self).__init__()
        self.ls_mask_filename = path.join(self.ls_masks_path,
                                    "icon_r2b4_013_0031_mask_downscaled_to_10min_latlon_corrected_with_grid.nc")
        self.output_rdirs_filename = "/Users/thomasriddick/Documents/data/temp/rdirs_r2b4_comb_trial.nc"
        self.output_catchments_filename = "/Users/thomasriddick/Documents/data/temp/catchment_r2b4_comb_trial.nc"
        self.output_cumulative_flow_filename = "/Users/thomasriddick/Documents/data/temp/accf_r2b4_comb_trial.nc"
        self.working_directory = "/Users/thomasriddick/Documents/data/temp"
        self.\
        no_intermediaries_combine_hydrosheds_plus_rdirs_from_corrected_orog_driver()

    def trial_generate_r2b5_mask_10min_combined_river_directions(self):
        super(Icon_Coarse_River_Directions_Creation_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.ls_mask_filename = path.join(self.ls_masks_path,
                                    "icon_r2b5_019_0032_mask_downscaled_to_10min_latlon_corrected_with_grid.nc")
        self.output_rdirs_filename = self.generated_rdir_filepath + file_label + ".nc"
        self.output_catchments_filename = self.generated_catchments_path + file_label + '.nc'
        self.output_cumulative_flow_filename = self.generated_flowmaps_filepath + file_label + '.nc'
        self.working_directory = "/Users/thomasriddick/Documents/data/temp"
        self.\
        no_intermediaries_combine_hydrosheds_plus_rdirs_from_corrected_orog_driver()


    def _read_and_validate_config(self):
        """Reads and checks format of config file

        Arguments: None
        Returns: ConfigParser object; the read and checked configuration
        """

        valid_config = True
        config = ConfigParser.ConfigParser()
        print "Read python driver options from file {0}".format(self.python_config_filename)
        config.read(self.python_config_filename)
        valid_config = valid_config \
            if config.has_section("input_options") else False
        valid_config = valid_config \
            if config.has_option("input_options",
                                 "ten_minute_corrected_orography_filename") else False
        valid_config = valid_config \
            if config.has_option("input_options",
                                 "ten_minute_hydrosheds_au_auf_sa_river_directions_filename") else False
        if not valid_config:
            raise RuntimeError("Invalid configuration file supplied")
        if not config.has_section("input_fieldname_options"):
            config.add_section("input_fieldname_options")
        if not config.has_option("input_fieldname_options","input_truesinks_fieldname"):
            config.set("input_fieldname_options","input_truesinks_fieldname","truesinks")
        if not config.has_option("input_fieldname_options","input_landsea_mask_fieldname"):
            config.set("input_fieldname_options","input_landsea_mask_fieldname","lsm")
        if not config.has_option("input_fieldname_options",
                                 "ten_minute_hydrosheds_au_auf_sa_river_directions_fieldname"):
            config.set("input_fieldname_options",
                       "ten_minute_hydrosheds_au_auf_sa_river_directions_fieldname","rdirs")
        if not config.has_option("input_fieldname_options",
                                 "river_directions_to_reprocess_fieldname"):
            config.set("input_fieldname_options",
                       "river_directions_to_reprocess_fieldname","rdirs")
        if not config.has_section("output_fieldname_options"):
            config.add_section("output_fieldname_options")
        if not config.has_option("output_fieldname_options","output_river_directions_fieldname"):
            config.set("output_fieldname_options","output_river_directions_fieldname","rdirs")
        if not config.has_option("output_fieldname_options","output_catchments_fieldname"):
            config.set("output_fieldname_options","output_catchments_fieldname","catchments")
        if not config.has_option("output_fieldname_options","output_cumulative_flow_fieldname"):
            config.set("output_fieldname_options","output_cumulative_flow_fieldname","acc")
        if not config.has_section("general_options"):
            config.add_section("general_options")
        if not config.has_option("general_options","keep_all_internal_basins"):
            config.set("general_options","keep_all_internal_basins","False")
        if not config.has_option("general_options",
                                 "replace_internal_basins_with_rdirs_with_truesinks"):
            config.set("general_options",
                       "replace_internal_basins_with_rdirs_with_truesinks","False")
        if not config.has_option("general_options","replace_only_catchments"):
            config.set("general_options","replace_only_catchments","")
        if not config.has_option("general_options","exclude_catchments"):
            config.set("general_options","exclude_catchments","")
        return config

    def no_intermediaries_combine_hydrosheds_plus_rdirs_from_corrected_orog_driver(self):
        """Combines Hydrosheds river direction with those from a corrected orography and possibly removes sinks"""
        config = self._read_and_validate_config()
        final_loops_log_filename =  path.splitext(self.output_catchments_filename)[0] + '_loops.log'
        if self.rerun_post_processing_only is not None:
          final_rdirs = \
            iodriver.advanced_field_loader(self.rerun_post_processing_only,
                                           field_type="RiverDirections",
                                           fieldname=config.get("input_fieldname_options",
                                           "river_directions_to_reprocess_fieldname"))
        else:
          orography_filename = config.get("input_options","ten_minute_corrected_orography_filename")
          orography = dynamic_hd.load_field(orography_filename,
                                            file_type=dynamic_hd.get_file_extension(orography_filename),
                                            field_type='Orography', grid_type="LatLong10min")
          hydrosheds_rdirs_au_af_sa_10min_filename = \
            config.get("input_options","ten_minute_hydrosheds_au_auf_sa_river_directions_filename")
          hydrosheds_rdirs = \
            iodriver.advanced_field_loader(hydrosheds_rdirs_au_af_sa_10min_filename,
                                           field_type="RiverDirections",
                                           fieldname=config.get("input_fieldname_options",
                                           "ten_minute_hydrosheds_au_auf_sa_river_directions_fieldname"))
          second_intermediate_loops_log_filename = tempfile.mkstemp(suffix=".txt",
                                                                    prefix="loops_log_second_int",
                                                                    dir="")[1]
          orography.flip_data_ud()
          orography.rotate_field_by_a_hundred_and_eighty_degrees()
          truesinks_dummy = field.makeEmptyField("Generic",np.bool_,grid_type="LatLong10min")
          truesinks_dummy.set_all(False)
          if self.true_sinks_filename is not None:
            use_true_sinks = True
            truesinks = iodriver.advanced_field_loader(self.true_sinks_filename,
                                                       field_type="Generic",
                                                       fieldname=config.get("input_fieldname_options",
                                                                            "input_truesinks_fieldname"))
          else:
            use_true_sinks = False
            if config.getboolean("general_options",
                                 "replace_internal_basins_with_rdirs_with_truesinks"):
              warnings.warn("Option replace_internal_basins_with_rdirs_with_truesinks "
                            "ignored when no true sinks file is specified")
          first_intermediate_rdirs = field.makeEmptyField("RiverDirections",np.float64,grid_type="LatLong10min")
          if use_true_sinks:
            first_intermediate_rdirs_no_sinks = field.makeEmptyField("RiverDirections",np.float64,
                                                                     grid_type="LatLong10min")
          ls_mask = iodriver.advanced_field_loader(self.ls_mask_filename,field_type="Generic",
                                                   fieldname=config.get("input_fieldname_options",
                                                                        "input_landsea_mask_fieldname"))
          ls_mask.set_data(np.ascontiguousarray(ls_mask.get_data(),
                                                dtype=np.int32))
          next_cell_lat_index_in_dummy = np.zeros(ls_mask.get_data().shape,dtype=np.int32,order='C')
          next_cell_lon_index_in_dummy = np.zeros(ls_mask.get_data().shape,dtype=np.int32,order='C')
          catchment_nums_dummy = np.zeros(ls_mask.get_data().shape,dtype=np.int32,order='C')
          fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=
                                                 np.ascontiguousarray(orography.get_data(), #@UndefinedVariable
                                                                      dtype=np.float64),
                                                 method = 4,
                                                 use_ls_mask = True,
                                                 landsea_in = np.ascontiguousarray(ls_mask.get_data(),
                                                                                   dtype=np.int32),
                                                 set_ls_as_no_data_flag = False,
                                                 use_true_sinks = False,
                                                 true_sinks_in = np.ascontiguousarray(truesinks_dummy.\
                                                                                      get_data(),
                                                                                       dtype=np.int32),
                                                 next_cell_lat_index_in = next_cell_lat_index_in_dummy,
                                                 next_cell_lon_index_in = next_cell_lon_index_in_dummy,
                                                 rdirs_in =
                                                 first_intermediate_rdirs.get_data() if not use_true_sinks else
                                                 first_intermediate_rdirs_no_sinks.get_data(),
                                                 catchment_nums_in = catchment_nums_dummy,
                                                 prefer_non_diagonal_initial_dirs = False)
          if use_true_sinks:
            fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=
                                                   np.ascontiguousarray(orography.get_data(), #@UndefinedVariable
                                                                        dtype=np.float64),
                                                   method = 4,
                                                   use_ls_mask = True,
                                                   landsea_in = np.ascontiguousarray(ls_mask.get_data(),
                                                                                     dtype=np.int32),
                                                   set_ls_as_no_data_flag = False,
                                                   use_true_sinks = True,
                                                   true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                         dtype=np.int32),
                                                   next_cell_lat_index_in = next_cell_lat_index_in_dummy,
                                                   next_cell_lon_index_in = next_cell_lon_index_in_dummy,
                                                   rdirs_in =
                                                   first_intermediate_rdirs.get_data(),
                                                   catchment_nums_in = catchment_nums_dummy,
                                                   prefer_non_diagonal_initial_dirs = False)
          second_intermediate_rdirs = utilities.splice_rdirs(rdirs_matching_ls_mask=first_intermediate_rdirs,
                                                             ls_mask=ls_mask,
                                                             other_rdirs=hydrosheds_rdirs)
          second_intermediate_catchments = comp_catchs.compute_catchments_cpp(second_intermediate_rdirs.get_data(),
                                                                              second_intermediate_loops_log_filename)
          second_intermediate_catchments = field.Field(comp_catchs.\
                                                       renumber_catchments_by_size(second_intermediate_catchments,
                                                                                   second_intermediate_loops_log_filename),
                                                       grid="LatLong10min")
          if config.getboolean("general_options","keep_all_internal_basins"):
            third_intermediate_rdirs = second_intermediate_rdirs
          else:
            third_intermediate_rdirs = \
              utilities.remove_endorheic_basins(rdirs=second_intermediate_rdirs,
                                                catchments=second_intermediate_catchments,
                                                rdirs_without_endorheic_basins=
                                                 first_intermediate_rdirs_no_sinks if
                                                 (use_true_sinks and not config.getboolean('general_options',
                                                  'replace_internal_basins_with_rdirs_with_truesinks'))
                                                else first_intermediate_rdirs,
                                                replace_only_catchments=([int(value) for value in
                                                                         config.get('general_options',
                                                                                    'replace_only_catchments').\
                                                                         split(",")]
                                                                         if config.get('general_options',
                                                                                       'replace_only_catchments')
                                                                         else []),
                                                exclude_catchments=([int(value) for value in
                                                                    config.get('general_options',
                                                                               'exclude_catchments').\
                                                                         split(",")]
                                                                         if config.get('general_options',
                                                                                       'replace_only_catchments')
                                                                         else []) )
          third_intermediate_flowtocell = field.\
            CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=third_intermediate_rdirs.get_data(),
                                                               lsmask=None,
                                                               use_f2py_func=True,
                                                               use_f2py_sparse_iterator=True,
                                                               nlat=1080,nlong=2160),
                                                               grid='LatLong10min')
          third_intermediate_rdirs.make_contiguous()
          third_intermediate_flowtocell.make_contiguous()
          first_intermediate_rdirs.make_contiguous()
          if use_true_sinks:
            first_intermediate_rdirs_no_sinks.make_contiguous()
          final_rdirs = utilities.replace_streams_downstream_from_loop(third_intermediate_rdirs,
                                                             cumulative_flow=third_intermediate_flowtocell,
                                                             other_rdirs=
                                                             first_intermediate_rdirs if not use_true_sinks else
                                                             first_intermediate_rdirs_no_sinks)
        final_catchments = comp_catchs.compute_catchments_cpp(final_rdirs.get_data(),
                                                              final_loops_log_filename)
        final_catchments = field.Field(comp_catchs.\
                                       renumber_catchments_by_size(final_catchments,
                                                                   final_loops_log_filename),
                                       grid="LatLong10min")
        final_flowtocell = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=final_rdirs.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=1080,nlong=2160),
                                                                                    grid='LatLong10min')
        if self.rerun_post_processing_only is None:
          iodriver.advanced_field_writer(self.output_rdirs_filename,final_rdirs,
                                         fieldname=config.get("output_fieldname_options",
                                                              "output_river_directions_fieldname"))
        iodriver.advanced_field_writer(self.output_catchments_filename,final_catchments,
                                       fieldname=config.get("output_fieldname_options",
                                                            "output_catchments_fieldname"))
        iodriver.advanced_field_writer(self.output_cumulative_flow_filename,
                                       final_flowtocell,
                                       fieldname=config.get("output_fieldname_options",
                                                            "output_cumulative_flow_fieldname"))
        if self.rerun_post_processing_only is None:
          os.remove(second_intermediate_loops_log_filename)

def setup_and_run_icon_para_gen_from_command_line_arguments(args):
    """Setup and run the icon hd parameter generation code from the command line arguments passed in by main"""
    driver_object = Icon_Coarse_River_Directions_Creation_Drivers(**vars(args))
    driver_object.no_intermediaries_combine_hydrosheds_plus_rdirs_from_corrected_orog_driver()

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
    parser.add_argument('ls_mask_filename',
                        metavar='ls-mask-filename',
                        help='Full path to land-sea mask to use',
                        type=str)
    parser.add_argument('output_rdirs_filename',
                        metavar='output-rdirs-filename',
                        help='Full path to target output river direction file',
                        type=str)
    parser.add_argument('output_catchments_filename',
                        metavar='output-catchments-filename',
                        help='Full path to target output catchment file',
                        type=str)
    parser.add_argument('output_cumulative_flow_filename',
                        metavar='output-cumulative-flow-filename',
                        help='Full path to target output cumulative flow file',
                        type=str)
    parser.add_argument('python_config_filename',
                        metavar='python-config-filename',
                        help='Full path to python configuration file')
    parser.add_argument('working_directory',
                        metavar='working-directory',
                        help='Full path to working directory')
    parser.add_argument('-t','--true_sinks_filename',
                        metavar='true-sinks-filename',
                        help='Full path to true sink file',
                        type=str,
                        default=None)
    parser.add_argument('-r','--rerun-post-processing-only',
                        metavar='rerun-post-processing-only',
                        help='Skip the main script and just run post processing',
                        type=str,
                        default=None)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_icon_para_gen_from_command_line_arguments(args)
