'''
Created on Feb 2, 2018

@author: thomasriddick
'''

import argparse
import os.path as path
from Dynamic_HD_Scripts.command_line_drivers import setup_validator
from Dynamic_HD_Scripts.tools import cotat_plus_driver
from Dynamic_HD_Scripts.tools import loop_breaker_driver
from Dynamic_HD_Scripts.tools import upscale_orography_driver
from Dynamic_HD_Scripts.tools import fill_sinks_driver
from Dynamic_HD_Scripts.tools import compute_catchments
from Dynamic_HD_Scripts.tools import create_connected_lsmask_driver
from Dynamic_HD_Scripts.tools import create_orography_driver
from Dynamic_HD_Scripts.tools import determine_river_directions
from Dynamic_HD_Scripts.tools import flow_to_grid_cell
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers import dynamic_hd_driver

def get_option_if_defined(config,section,option):
    return (config.get(section,option)
            if config.has_option(section,option) else None)

def get_boolean_option_if_defined(config,section,option,default):
    return (config.getboolean(section,option)
            if config.has_option(section,option) else default)

class HDOperatorDrivers(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def primary_driver(self,configuration_filepath):
        setup_validator_object = setup_validator.SetupValidator()
        setup_validator_object.read_config(configuration_filepath)
        driver_to_use = setup_validator_object.process_config()
        config = setup_validator_object.get_config()
        getattr(self,driver_to_use)(config)

    def print_driver_options(self,driver_option_printout_target_filepath):
        setup_validator_object = setup_validator.SetupValidator()
        with open(driver_option_printout_target_filepath,'w') as f:
            f.write(setup_validator_object.print_valid_options())

    def cotat_plus_driver(self,config):
        cotat_plus_driver.advanced_cotat_plus_driver(input_fine_rdirs_filepath=\
                                                     config.get("input_filepaths",
                                                                        "fine_rdirs"),
                                                     input_fine_total_cumulative_flow_path=\
                                                     config.get("input_filepaths",
                                                                        "fine_cumulative_flow"),
                                                     output_course_rdirs_filepath=\
                                                     config.get("output_filepaths",
                                                                        "coarse_rdirs_out"),
                                                     input_fine_rdirs_fieldname=\
                                                     config.get("input_fieldnames",
                                                                        "fine_rdirs"),
                                                     input_fine_total_cumulative_flow_fieldname=\
                                                     config.get("input_fieldnames",
                                                                        "fine_cumulative_flow"),
                                                     output_course_rdirs_fieldname=\
                                                     config.get("output_fieldnames",
                                                                        "coarse_rdirs_out"),
                                                     cotat_plus_parameters_filepath=\
                                                     config.get("river_direction_upscaling",
                                                                        "parameters_filepath"),
                                                     scaling_factor=\
                                                     int(config.get("general",
                                                                    "upscaling_factor")))

    def loop_breaking_driver(self,config):
        loop_breaker_driver.advanced_loop_breaker_driver(input_course_rdirs_filepath=\
                                                         config.get("input_filepaths",
                                                                    "coarse_rdirs"),
                                                         input_course_cumulative_flow_filepath=\
                                                         config.get("input_filepaths",
                                                                    "coarse_cumulative_flow"),
                                                         input_course_catchments_filepath=\
                                                         config.get("input_filepaths",
                                                                    "coarse_catchments"),
                                                         input_fine_rdirs_filepath=\
                                                         config.get("input_filepaths",
                                                                    "fine_rdirs"),
                                                         input_fine_cumulative_flow_filepath=\
                                                         config.get("input_filepaths",
                                                                    "fine_cumulative_flow"),
                                                         output_updated_course_rdirs_filepath=\
                                                         config.get("output_filepaths",
                                                                    "coarse_rdirs_out"),
                                                         input_course_rdirs_fieldname=\
                                                         config.get("input_fieldnames",
                                                                    "coarse_rdirs"),
                                                         input_course_cumulative_flow_fieldname=\
                                                         config.get("input_fieldnames",
                                                                    "coarse_cumulative_flow"),
                                                         input_course_catchments_fieldname=\
                                                         config.get("input_fieldnames",
                                                                    "coarse_catchments"),
                                                         input_fine_rdirs_fieldname=\
                                                         config.get("input_fieldnames",
                                                                    "fine_rdirs"),
                                                         input_fine_cumulative_flow_fieldname=\
                                                         config.get("input_fieldnames",
                                                                    "fine_cumulative_flow"),
                                                         output_updated_course_rdirs_fieldname=\
                                                         config.get("output_fieldnames",
                                                                    "coarse_rdirs_out"),
                                                         loop_nums_list_filepath=\
                                                         config.get("input_filepaths",
                                                                    "loop_logfile"),
                                                         scaling_factor=\
                                                         int(config.get("general",
                                                                        "upscaling_factor")))

    def orography_upscaling_driver(self,config):
        upscale_orography_driver.advanced_drive_orography_upscaling(input_fine_orography_file=\
                                                                    config.get("input_filepaths",
                                                                                       "fine_orography"),
                                                                    output_course_orography_file=\
                                                                    config.get("output_filepaths",
                                                                                       "coarse_orography"),
                                                                    input_orography_fieldname=\
                                                                    config.get("input_fieldnames",
                                                                                       "fine_orography"),
                                                                    output_course_orography_fieldname=\
                                                                    config.get("output_fieldnames",
                                                                                       "coarse_orography"),
                                                                    landsea_file=\
                                                                    get_option_if_defined(config,
                                                                                          "input_filepaths",
                                                                                          "fine_landsea"),
                                                                    true_sinks_file=\
                                                                    get_option_if_defined(config,
                                                                                          "input_filepaths",
                                                                                          "fine_truesinks"),
                                                                    landsea_fieldname=\
                                                                    get_option_if_defined(config,
                                                                                          "input_fieldname",
                                                                                          "fine_landsea"),
                                                                    true_sinks_fieldname=\
                                                                    get_option_if_defined(config,
                                                                                          "input_fieldname",
                                                                                          "fine_truesinks"),
                                                                    upscaling_parameters_filename=\
                                                                    config.get("orography_upscaling",
                                                                                       "parameter_filepath"),
                                                                    scaling_factor=\
                                                                    int(config.get("general",
                                                                                       "upscaling_factor")))

    def sink_filling_driver(self,config):
        fill_sinks_driver.generate_orography_with_sinks_filled_advanced_driver(input_orography_filename=\
                                                                               config.get("input_filepaths",
                                                                                          "orography"),
                                                                               output_orography_filename=\
                                                                               config.get("output_filepaths",
                                                                                          "orography_out"),
                                                                               input_orography_fieldname=\
                                                                               config.get("input_fieldnames",
                                                                                          "orography"),
                                                                               output_orography_fieldname=\
                                                                               config.get("output_fieldnames",
                                                                                          "orography_out"),
                                                                               ls_mask_filename=\
                                                                               get_option_if_defined(config,
                                                                                                     "input_filepaths",
                                                                                                     "landsea"),
                                                                               truesinks_filename=\
                                                                               get_option_if_defined(config,
                                                                                                     "input_filepaths",
                                                                                                     "truesinks"),
                                                                               ls_mask_fieldname=\
                                                                               get_option_if_defined(config,
                                                                                                     "input_fieldnames",
                                                                                                     "landsea"),
                                                                               truesinks_fieldname=\
                                                                               get_option_if_defined(config,
                                                                                                     "input_fieldnames",
                                                                                                     "truesinks"),
                                                                               add_slight_slope_when_filling_sinks=\
                                                                               config.getboolean("sink_filling",
                                                                                                 "add_slight_slope"
                                                                                                 "_when_filling_"
                                                                                                 "sinks"),
                                                                               slope_param=
                                                                               float(config.get("sink_filling",
                                                                                     "slope_param")))

    def river_carving_driver(self,config):
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=\
                                                                      config.get("input_filepaths",
                                                                                         "orography"),
                                                                      output_filename=\
                                                                      config.get("output_filepaths",
                                                                                         "rdirs_out"),
                                                                      fieldname=\
                                                                      config.get("input_fieldnames",
                                                                                         "orography"),
                                                                      output_fieldname=\
                                                                      config.get("output_fieldnames",
                                                                                         "rdirs_out"),
                                                                      ls_mask_filename=\
                                                                      get_option_if_defined(config,
                                                                                            "input_filepaths",
                                                                                            "landsea"),
                                                                      truesinks_filename=\
                                                                      get_option_if_defined(config,
                                                                                            "input_filepaths",
                                                                                            "truesinks"),
                                                                      catchment_nums_filename=\
                                                                      get_option_if_defined(config,
                                                                                            "input_filepaths",
                                                                                            "catchments_out"),
                                                                      ls_mask_fieldname=\
                                                                      get_option_if_defined(config,
                                                                                            "input_fieldnames",
                                                                                            "landsea"),
                                                                      truesinks_fieldname=\
                                                                      get_option_if_defined(config,
                                                                                            "input_fieldnames",
                                                                                            "truesinks"),
                                                                      catchment_nums_fieldname=\
                                                                      get_option_if_defined(config,
                                                                                            "input_fieldnames",
                                                                                            "catchments_out"))

    def compute_catchment_driver(self,config):
        compute_catchments.advanced_main(filename=config.get("input_filepaths","rdirs"),
                                         fieldname=config.get("input_fieldnames","rdirs"),
                                         output_filename=config.get("output_filepaths",
                                                                            "catchments_out"),
                                         output_fieldname=config.get("output_fieldnames",
                                                                             "catchments_out"),
                                         loop_logfile=str(config.get("output_filepaths",
                                                                     "loop_logfile")))

    def compute_cumulative_flow_driver(self,config):
        flow_to_grid_cell.advanced_main(rdirs_filename=config.get("input_filepaths","rdirs"),
                                        output_filename=config.get("output_filepaths",
                                                                   "cumulative_flow_out"),
                                        rdirs_fieldname=config.get("input_fieldnames","rdirs"),
                                        output_fieldname=config.get("output_fieldnames",
                                                                    "cumulative_flow_out"))

    def connected_ls_mask_creation_driver(self,config):
        create_connected_lsmask_driver.\
            advanced_connected_lsmask_creation_driver(input_lsmask_filename=config.\
                                                      get("input_filepaths","landsea"),
                                                      output_lsmask_filename=config.\
                                                      get("output_filepaths","landsea_out"),
                                                      input_lsmask_fieldname=config.\
                                                      get("input_filepaths","landsea"),
                                                      output_lsmask_fieldname=config.\
                                                      get("output_filepaths","landsea_out"),
                                                      input_ls_seed_points_filename=config.\
                                                      get("input_filepaths","ls_seed_points"),
                                                      input_ls_seed_points_fieldname=config.\
                                                      get("input_fieldnames","ls_seed_points"),
                                                      input_ls_seed_points_list_filename=config.\
                                                      get("input_filenames","ls_seed_points_list"),
                                                      use_diagonals_in=config.\
                                                      getboolean("connected_lsmask_generation",
                                                                  "use_diagonals"),
                                                      rotate_seeds_about_polar_axis=config.\
                                                      getboolean("connected_lsmask_generation",
                                                                  "rotate_seeds_about_polar_axis"),
                                                      flip_seeds_ud=config.\
                                                      getboolean("connected_lsmask_generation",
                                                                  "flip_seeds_upside_down"))

    def orography_rebasing_driver(self,config):
        utilities.advanced_rebase_orography_driver(orography_filename=config.\
                                                   get("input_filepaths","orography"),
                                                   present_day_base_orography_filename=config.\
                                                   get("input_filepaths","present_day_base_orography"),
                                                   present_day_reference_orography_filename=config.\
                                                   get("input_filepaths","present_day_reference_orography"),
                                                   rebased_orography_filename=config.\
                                                   get("output_filepaths","orography_out"),
                                                   orography_fieldname=config.\
                                                   get("input_fieldnames","orography"),
                                                   present_day_base_orography_fieldname=config.\
                                                   get("input_fieldnames","present_day_base_orography"),
                                                   present_day_reference_orography_fieldname=config.\
                                                   get("input_fieldnames","present_day_reference_orography"),
                                                   rebased_orography_fieldname=config.\
                                                   get("output_fieldnames","orography_out"))

    def orography_correction_application_driver(self,config):
        utilities.advanced_apply_orog_correction_field(original_orography_filename=config.\
                                                       get("input_filepaths","orography"),
                                                       orography_corrections_filename=config.\
                                                       get("input_filepaths","orography_corrections"),
                                                       corrected_orography_filename=config.\
                                                       get("output_filepaths","orography_out"),
                                                       original_orography_fieldname=config.\
                                                       get("input_fieldnames","orography"),
                                                       orography_corrections_fieldname=config.\
                                                       get("input_fieldnames","orography_corrections"),
                                                       corrected_orography_fieldname=config.\
                                                       get("output_fieldnames","orography_out"))

    def orography_correction_generation_driver(self,config):
        utilities.advanced_orog_correction_field_generator(original_orography_filename=config.\
                                                           get("input_filepaths",
                                                               "original_orography"),
                                                           corrected_orography_filename=config.\
                                                           get("input_filepaths",
                                                               "corrected_orography"),
                                                           orography_corrections_filename=config.\
                                                           get("output_filepaths",
                                                               "orography_corrections"),
                                                           original_orography_fieldname=config.\
                                                           get("input_fieldnames",
                                                               "original_orography"),
                                                           corrected_orography_fieldname=config.\
                                                           get("input_fieldnames",
                                                               "corrected_orography"),
                                                           orography_corrections_fieldname=config.\
                                                           get("output_fieldnames",
                                                               "orography_corrections"))

    def corrected_and_tarasov_upscaled_orography_merging_driver(self,config):
        utilities.advanced_merge_corrected_and_tarasov_upscaled_orography(input_corrected_orography_file=config.\
                                                                          get("input_filepaths",
                                                                              "corrected_orography"),
                                                                          input_tarasov_upscaled_orography_file=config.\
                                                                          get("input_filepaths",
                                                                              "upscaled_orography"),
                                                                          output_merged_orography_file=config.\
                                                                          get("output_filepaths",
                                                                              "orography_out"),
                                                                          input_corrected_orography_fieldname=config.\
                                                                          get("input_fieldnames",
                                                                              "corrected_orography"),
                                                                          input_tarasov_upscaled_orography_fieldname=config.\
                                                                          get("input_fieldnames",
                                                                              "upscaled_orography"),
                                                                          output_merged_orography_fieldname=config.\
                                                                          get("output_filepaths",
                                                                              "orography_out"),
                                                                          use_upscaled_orography_only_in_region=config.\
                                                                          get("upscale_orography_merging",
                                                                              "use_upscaled_orography_only_in_region") if
                                                                          config.has_option("upscale_orography_merging",
                                                                                            "use_upscaled_orography_"
                                                                                            "only_in_region") else None)

    def replace_corrected_orog_with_orig_for_glcted_grid_points_drivers(self,config):
        utilities.\
        advanced_replace_corrected_orog_with_orig_for_glcted_grid_points_drivers(input_corrected_orography_file=\
                                                                                 config.get("input_filepaths",
                                                                                            "original_orography"),
                                                                                 input_original_orography_file=\
                                                                                 config.get("input_filepaths",
                                                                                            "corrected_orography"),
                                                                                 input_glacier_mask_file=\
                                                                                 config.get("input_filepaths",
                                                                                            "glacier_mask"),
                                                                                 out_orography_file=\
                                                                                 config.get("output_filepaths",
                                                                                            "orography_out"),
                                                                                 input_corrected_orography_fieldname=\
                                                                                 config.get("input_fieldnames",
                                                                                            "original_orography"),
                                                                                 input_original_orography_fieldname=\
                                                                                 config.get("input_fieldnames",
                                                                                            "corrected_orography"),
                                                                                 input_glacier_mask_fieldname=\
                                                                                 config.get("input_fieldnames",
                                                                                            "glacier_mask"),
                                                                                 out_orography_fieldname=\
                                                                                 config.get("output_fieldnames",
                                                                                            "orography_out"))
    def orography_creation_driver(self,config):
      create_orography_driver.\
        advanced_orography_creation_driver(landsea_mask_filename=config.\
                                           get("input_filepaths","landsea"),
                                           inclines_filename=config.\
                                           get("input_filepaths","inclines"),
                                           orography_filename=config.\
                                           get("output_filepaths","orography_out"),
                                           landsea_mask_fieldname=config.\
                                           get("input_fieldnames","landsea"),
                                           inclines_fieldname=config.\
                                           get("input_fieldnames","inclines"),
                                           orography_fieldname=config.\
                                           get("output_fieldnames","orography_out"))

    def river_direction_determination_driver(self,config):
      determine_river_directions.\
        advanced_river_direction_determination_driver(rdirs_filename=config.\
                                                      get("output_filepaths","rdirs_out"),
                                                      orography_filename=config.\
                                                      get("input_filepaths","orography"),
                                                      lsmask_filename=config.\
                                                      get("input_filepaths","landsea"),
                                                      truesinks_filename=\
                                                      get_option_if_defined(config,
                                                                            "input_filepaths",
                                                                            "truesinks"),
                                                      rdirs_fieldname=config.\
                                                      get("output_fieldnames","rdirs_out"),
                                                      orography_fieldname=config.\
                                                      get("input_fieldnames","orography"),
                                                      lsmask_fieldname=config.\
                                                      get("input_fieldnames","landsea"),
                                                      truesinks_fieldname=\
                                                      get_option_if_defined(config,
                                                                            "input_fieldnames",
                                                                            "truesinks"),
                                                      always_flow_to_sea=\
                                                      get_boolean_option_if_defined(config,
                                                                                    "river_direction_determination",
                                                                                    "always_flow_to_sea",
                                                                                    True),
                                                      use_diagonal_nbrs=\
                                                      get_boolean_option_if_defined(config,
                                                                                    "river_direction_determination",
                                                                                    "use_diagonal_nbrs",
                                                                                    True),
                                                      mark_pits_as_true_sinks=\
                                                      get_boolean_option_if_defined(config,
                                                                                    "river_direction_determination",
                                                                                    "mark_pits_as_true_sinks",
                                                                                    True))

    def parameter_generation_driver(self,config):
        rdirs_file=config.get("input_filepaths","rdirs")
        orography_file=config.get("input_filepaths","orography")
        ls_mask=config.get("input_filepaths","landsea")
        hdpara_filepath=config.get("output_filepaths","hdpara_out")
        ancillary_data_path=config.get("parameter_generation","ancillary_data_path")
        working_dir=config.get("parameter_generation","working_dir")
        dyn_hd_driver = dynamic_hd_driver.Dynamic_HD_Drivers()
        dyn_hd_driver._generate_flow_parameters(rdir_file=rdirs_file,
                                                topography_file=orography_file,
                                                inner_slope_file=\
                                                path.join(ancillary_data_path,'bin_innerslope.dat'),
                                                lsmask_file=ls_mask,
                                                null_file=\
                                                path.join(ancillary_data_path,'null.dat'),
                                                area_spacing_file=\
                                                path.join(ancillary_data_path,'fl_dp_dl.dat'),
                                                orography_variance_file=\
                                                path.join(ancillary_data_path,'bin_toposig.dat'),
                                                output_dir=path.join(working_dir,
                                                                     'hd_flow_params'),
                                                production_run=True)
        dyn_hd_driver._generate_hd_file(rdir_file=path.splitext(rdirs_file)[0] + ".dat",
                                        lsmask_file=path.splitext(ls_mask)[0] + ".dat",
                                        null_file=\
                                        path.join(ancillary_data_path,'null.dat'),
                                        area_spacing_file=\
                                        path.join(ancillary_data_path,
                                                  'fl_dp_dl.dat'),
                                        hd_grid_specs_file=\
                                        path.join(ancillary_data_path,
                                                  'grid_0_5.txt'),
                                        output_file=hdpara_filepath,
                                        paras_dir=path.join(working_dir,
                                                            'hd_flow_params'),
                                        production_run=True)

def setup_and_run_hd_operator_driver_from_command_line_arguments(args):
    driver_object = HDOperatorDrivers()
    if args.print_hd_driver_options:
        driver_object.print_driver_options(args.print_hd_driver_options)
    else:
        driver_object.primary_driver(args.run_hd_driver_script)

class Arguments(object):
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser("Run HD Operator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-r','--run',metavar='settings-ini-file',
                       dest='run_hd_driver_script',type=str,
                       help="Run the HD operator specified in supplied ini file")
    group.add_argument('-p','--print',metavar='target-destination-file',
                       dest='print_hd_driver_options',type=str,
                       help="Print the required and optional options for"
                            "each HD operator")
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_hd_operator_driver_from_command_line_arguments(args)
