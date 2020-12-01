'''
Created on April 1, 2020

@author: thomasriddick
'''
import re
import os
import determine_river_directions
import dynamic_hd_driver as dyn_hd_dr
import dynamic_lake_operators
import iodriver
import os.path as path
import ConfigParser
import numpy as np
import utilities
import field
import libs.fill_sinks_wrapper as fill_sinks_wrapper
import libs.lake_operators_wrapper as lake_operators_wrapper  #@UnresolvedImport
from timeit import default_timer as timer
import cdo
import argparse
from flow_to_grid_cell import create_hypothetical_river_paths_map
import compute_catchments as comp_catchs
from cotat_plus_driver import run_cotat_plus
from loop_breaker_driver import run_loop_breaker

class Dynamic_Lake_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):
    """A class with methods used for running a production run of the dynamic HD and Lake generation code"""

    def __init__(self,input_orography_filepath=None,input_ls_mask_filepath=None,
                 input_water_to_redistribute_filepath=None,
                 output_hdparas_filepath=None,output_lakeparas_filepath=None,
                 output_lakestart_filepath=None,ancillary_data_directory=None,
                 working_directory=None,output_hdstart_filepath=None,
                 present_day_base_orography_filepath=None,glacier_mask_filepath=None):
        """Class constructor.

        Deliberately does NOT call constructor of Dynamic_HD_Drivers so the many paths
        within the data directory structure used for offline runs is not initialized here
        """

        self.original_orography_filename=input_orography_filepath
        self.original_ls_mask_filename=input_ls_mask_filepath
        self.input_water_to_redistribute_filepath=input_water_to_redistribute_filepath
        self.output_hdparas_filepath=output_hdparas_filepath
        self.ancillary_data_path=ancillary_data_directory
        self.working_directory_path=working_directory
        self.output_hdstart_filepath=output_hdstart_filepath
        self.output_lakeparas_filepath = output_lakeparas_filepath
        self.output_lakestart_filepath = output_lakestart_filepath
        self.present_day_base_orography_filename=present_day_base_orography_filepath
        self.glacier_mask_filename=glacier_mask_filepath
        self.tarasov_based_orog_correction=True
        self.python_config_filename=path.join(self.ancillary_data_path,
                                              "dynamic_lake_production_driver.cfg")

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
            if config.has_section("output_options") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_corrected_orog") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_rdirs") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_catchments") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_flowtocell") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_flowtorivermouths") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_rdirs") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_flowtocell") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_flowtorivermouths") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_catchments") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_rdirs") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_unfilled_orog") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_filled_orog") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_flowtocell") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_flowtorivermouths") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_catchments") else False
        value_config = valid_config \
            if config.has_section("input_fieldname_options") else False
        value_config = valid_config \
            if config.has_option("input_fieldname_options","input_orography_fieldname") else False
        value_config = valid_config \
            if config.has_option("input_fieldname_options","input_landsea_mask_fieldname") else False
        value_config = valid_config \
            if config.has_option("input_fieldname_options","input_glacier_mask_fieldname") else False
        value_config = valid_config \
            if config.has_option("input_fieldname_options","input_base_present_day_orography_fieldname") else False
        value_config = value_config \
            if config.has_option("input_fieldname_options","input_reference_present_day_fieldname") else False
        value_config = value_config \
            if config.has_option("input_fieldname_options","input_orography_corrections_fieldname") else False
        if not valid_config:
            raise RuntimeError("Invalid configuration file supplied")
        if not config.has_section("general_options"):
            config.add_section("general_options")
        if not config.has_option("general_options","generate_flow_parameters"):
            config.set("general_options","generate_flow_parameters","True")
        if not config.has_option("general_options","print_timing_information"):
            config.set("general_options","print_timing_information","False")
        return config

    def no_intermediaries_dynamic_lake_driver(self):
        """Generates necessary files for runing a dynamic lake model

        Arguments: None
        Returns: nothing
        """

        config = self._read_and_validate_config()
        print_timing_info = config.getboolean("general_options","print_timing_information")
        if print_timing_info:
            start_time = timer()
        base_hd_restart_file = path.join(self.ancillary_data_path,"hd_restart_from_hd_file_ten_minute_data_from_virna_"
                                        "0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_"
                                        "135934_after_one_year_running.nc")
        ref_hd_paras_file = path.join(self.ancillary_data_path,"hd_file_ten_minute_data_from_virna_0k_ALG4_sinkless_no_"
                                      "true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_123858_to_use_as_"
                                      "hdparas_ref.nc")
        if self.present_day_base_orography_filename:
            present_day_reference_orography_filename = path.join(self.ancillary_data_path,
                                                                 "ice5g_v1_2_00_0k_10min.nc")
        orography_corrections_filename = path.join(self.ancillary_data_path,
                                                   "ice5g_0k_lake_corrs_no_intermediaries_lake_corrections_driver_20200726_181304.nc")
        #Change ls mask to correct type
        ls_mask_10min = iodriver.advanced_field_loader(self.original_ls_mask_filename,
                                                       field_type='Generic',
                                                       fieldname=config.get("input_fieldname_options",
                                                                            "input_landsea_mask_fieldname"),
                                                       adjust_orientation=True)
        ls_mask_10min.change_dtype(np.int32)
        #Add corrections to orography
        orography_10min = iodriver.advanced_field_loader(self.original_orography_filename,
                                                         fieldname=config.get("input_fieldname_options",
                                                                              "input_orography_fieldname"),
                                                         field_type='Orography')
        if self.present_day_base_orography_filename:
            present_day_base_orography = \
            iodriver.advanced_field_loader(self.present_day_base_orography_filename,
                                           field_type='Orography',
                                           fieldname=config.get("input_fieldname_options",
                                                                "input_base_present_day_orography_fieldname"))
            present_day_reference_orography = \
            iodriver.advanced_field_loader(present_day_reference_orography_filename,
                                           field_type='Orography',
                                           fieldname=config.get("input_fieldname_options",
                                                                "input_reference_present_day_fieldname"))
            orography_10min = utilities.rebase_orography(orography=orography_10min,
                                                         present_day_base_orography=\
                                                         present_day_base_orography,
                                                         present_day_reference_orography=\
                                                         present_day_reference_orography)
        orography_corrections_10min =  iodriver.advanced_field_loader(orography_corrections_filename,
                                                                      fieldname=config.get("input_fieldname_options",
                                                                                           "input_orography_corrections_fieldname"),
                                                                      field_type='Orography')
        orography_uncorrected_10min = orography_10min.copy()
        orography_10min.add(orography_corrections_10min)
        truesinks = field.Field(np.empty((1,1),dtype=np.int32),grid='HD')
        if self.glacier_mask_filename:
            glacier_mask_10min = iodriver.advanced_field_loader(self.glacier_mask_filename,
                                                                fieldname=config.get("input_fieldname_options",
                                                                                     "input_glacier_mask_fieldname"),
                                                                field_type='Orography')
            orography_10min = utilities.\
            replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography=\
                                                                                orography_10min,
                                                                                input_original_orography=\
                                                                                orography_uncorrected_10min,
                                                                                input_glacier_mask=
                                                                                glacier_mask_10min)
            orography_10min.change_dtype(np.float64)
            orography_10min.make_contiguous()
            inverted_glacier_mask_10min = glacier_mask_10min.copy()
            inverted_glacier_mask_10min.invert_data()
            fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=
                                                   orography_10min.get_data(),
                                                   method = 1,
                                                   use_ls_mask = True,
                                                   landsea_in =
                                                   np.ascontiguousarray(inverted_glacier_mask_10min.get_data(),
                                                                        dtype=np.int32),
                                                   set_ls_as_no_data_flag = False,
                                                   use_true_sinks = False,
                                                   true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                        dtype=np.int32),
                                                   add_slope = False,epsilon = 0.0)
        if config.getboolean("output_options","output_corrected_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_corrected_orog.nc"),
                                           orography_10min,
                                           fieldname="z")#config.get("output_fieldname_options",
                                                     #           "output_10min_corrected_orog_fieldname"))
        #Generate orography with filled sinks
        if print_timing_info:
            time_before_sink_filling = timer()
        grid_dims_10min=orography_10min.get_grid().get_grid_dimensions()
        orography_10min_filled = orography_10min.copy()
        fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=
                                               np.ascontiguousarray(orography_10min_filled.get_data(),
                                                                                    dtype=np.float64),
                                               method = 1,
                                               use_ls_mask = True,
                                               landsea_in = np.ascontiguousarray(ls_mask_10min.get_data(),
                                                                                 dtype=np.int32),
                                               set_ls_as_no_data_flag = False,
                                               use_true_sinks = False,
                                               true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                    dtype=np.int32),
                                               add_slope = False,epsilon = 0.0)
        #Filter unfilled orography
        orography_10min = \
          field.Field(np.ascontiguousarray(orography_10min.get_data(),
                                           dtype=np.float64),
                      grid=orography_10min.get_grid())
        lake_operators_wrapper.filter_out_shallow_lakes(orography_10min.get_data(),
                                                        np.ascontiguousarray(orography_10min_filled.\
                                                                             get_data(),
                                                                             dtype=np.float64),
                                                        minimum_depth_threshold=5.0)
        orography_10min = \
            field.Field(dynamic_lake_operators.\
                        filter_narrow_lakes(input_unfilled_orography=
                                            orography_10min.get_data(),
                                            input_filled_orography=
                                            np.ascontiguousarray(orography_10min_filled.\
                                                                 get_data(),
                                                                 dtype=np.float64),
                                            interior_cell_min_masked_neighbors=5,
                                            edge_cell_max_masked_neighbors=4,
                                            max_range=5,
                                            iterations=5),
                        grid=orography_10min.get_grid())
        #Generate River Directions
        rdirs_10min = determine_river_directions.determine_river_directions(orography=orography_10min,
                                                                            lsmask=ls_mask_10min,
                                                                            truesinks=None,
                                                                            always_flow_to_sea=True,
                                                                            use_diagonal_nbrs=True,
                                                                            mark_pits_as_true_sinks=True)
        if config.getboolean("output_options","output_fine_rdirs"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "10min_rdirs.nc"),
                                           rdirs_10min,
                                           fieldname="rdirs")#config.get("output_fieldname_options",
                                                             #   "output_10min_rdirs_fieldname")
        #Run post processing
        if print_timing_info:
            time_before_10min_post_processing = timer()
        nlat10,nlong10 = grid_dims_10min
        flowtocell_10min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_10min.get_data(),
                                                                                    lsmask=ls_mask_10min.get_data(),
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat10,
                                                                                    nlong=nlong10),
                                                                                    grid='LatLong10min')
        rdirs_10min.mark_river_mouths(ls_mask_10min.get_data())
        rivermouths_10min = field.makeField(rdirs_10min.get_river_mouths(),'Generic','LatLong10min')
        flowtorivermouths_10min = field.makeField(flowtocell_10min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_10min.\
                                                                                  get_data()),
                                                  'Generic','LatLong10min')
        loops_log_10min_filename = path.join(self.working_directory_path,"loops_10min.log")
        catchments_10min = comp_catchs.compute_catchments_cpp(rdirs_10min.get_data(),
                                                              loops_log_10min_filename)
        catchments_10min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_10min,
                                                                               loops_log_10min_filename),
                                       grid=rdirs_10min.get_grid())
        if config.getboolean("output_options","output_fine_flowtocell"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_flowtocell.nc"),
                                           flowtocell_10min,
                                           fieldname="acc")#config.get("output_fieldname_options",
                                                        #        "output_10min_flow_to_cell"))
        if config.getboolean("output_options","output_fine_flowtorivermouths"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_flowtorivermouths.nc"),
                                           flowtorivermouths_10min,
                                           fieldname="accrm")#config.get("output_fieldname_options",
                                                              #"output_10min_flow_to_river_mouths"))
        if config.getboolean("output_options","output_fine_catchments"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_catchments.nc"),
                                           catchments_10min,
                                           fieldname="catch")#config.get("output_fieldname_options",
                                                      #          "output_10min_catchments"))
        #Run Upscaling
        if print_timing_info:
            time_before_upscaling = timer()
        loops_log_30min_filename = path.join(self.working_directory_path,"loops_30min.log")
        catchments_log_filename= path.join(self.working_directory_path,"catchments.log")
        cotat_plus_parameters_filename = path.join(self.ancillary_data_path,'cotat_plus_standard_params.nl')
        rdirs_30min = run_cotat_plus(rdirs_10min, flowtocell_10min,
                                      cotat_plus_parameters_filename,'HD')
        if config.getboolean("output_options","output_pre_loop_removal_course_rdirs"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_pre_loop_removal_rdirs.nc"),
                                           rdirs_30min,
                                           fieldname="rdirs")#config.get("output_fieldname_options",
                                                     #           "output_30min_pre_loop_removal_rdirs"))
        #Post processing
        if print_timing_info:
            time_before_30min_post_processing_one = timer()
        nlat30,nlong30 = rdirs_30min.get_grid().get_grid_dimensions()
        flowtocell_30min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_30min.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat30,
                                                                                    nlong=nlong30),
                                                                                    grid='HD')
        catchments_30min = comp_catchs.compute_catchments_cpp(rdirs_30min.get_data(),
                                                              loops_log_30min_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_30min_filename),
                                       grid="HD")
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic','HD')
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic','HD')
        if config.getboolean("output_options","output_pre_loop_removal_course_flowtocell"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "30min_pre_loop_removal_flowtocell.nc"),
                                           flowtocell_30min,
                                           fieldname="acc")#config.get("output_fieldname_options",
                                                     #           "output_30min_pre_loop_removal_flow_to_cell"))
        if config.getboolean("output_options","output_pre_loop_removal_course_flowtorivermouths"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_pre_loop_removal_flowtorivermouths.nc"),
                                             flowtorivermouths_30min,
                                             fieldname="accrm")#config.get("output_fieldname_options",
                                                               #   "output_30min_pre_loop_removal_flow_to_river_mouth"))
        if config.getboolean("output_options","output_pre_loop_removal_course_catchments"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_pre_loop_removal_catchments.nc"),
                                             catchments_30min,
                                             fieldname="catch")#config.get("output_fieldname_options",
                                                       #           "output_30min_pre_loop_removal_catchments"))
        #Run Loop Breaker
        if print_timing_info:
            time_before_loop_breaker = timer()
        loop_nums_list = []
        first_line_pattern = re.compile(r"^Loops found in catchments:$")
        with open(loops_log_30min_filename,'r') as f:
            if not first_line_pattern.match(f.readline().strip()):
                raise RuntimeError("Format of the file with list of catchments to remove loops from"
                                   " is invalid")
            for line in f:
                loop_nums_list.append(int(line.strip()))
        print 'Removing loops from catchments: ' + ", ".join(str(value) for value in loop_nums_list)
        rdirs_30min = run_loop_breaker(rdirs_30min,flowtocell_30min,
                                       catchments_30min,rdirs_10min,
                                       flowtocell_10min,loop_nums_list,
                                       course_grid_type="HD")
        if config.getboolean("output_options","output_course_rdirs"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_rdirs.nc"),
                                             rdirs_30min,
                                             fieldname="rdirs")#config.get("output_fieldname_options",
                                                        #          "output_30min_rdirs"))
        #Upscale the orography to the HD grid for calculating the flow parameters
        orography_30min= utilities.upscale_field(orography_10min,"HD","Sum",{},
                                                 scalenumbers=True)
        if config.getboolean("output_options","output_course_unfilled_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_unfilled_orog.nc"),
                                             orography_30min,
                                             fieldname="z")#config.get("output_fieldname_options",
                                                        #          "output_30min_unfilled_orog"))
        #Extract HD ls mask from river directions
        ls_mask_30min = field.RiverDirections(rdirs_30min.get_lsmask(),grid='HD')
        #Fill HD orography for parameter generation
        if print_timing_info:
            time_before_sink_filling = timer()
        truesinks = field.Field(np.empty((1,1),dtype=np.int32),grid='HD')
        fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=np.ascontiguousarray(orography_30min.get_data(), #@UndefinedVariable
                                                                                    dtype=np.float64),
                                               method = 1,
                                               use_ls_mask = True,
                                               landsea_in = np.ascontiguousarray(ls_mask_30min.get_data(),
                                                                             dtype=np.int32),
                                               set_ls_as_no_data_flag = False,
                                               use_true_sinks = False,
                                               true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                dtype=np.int32),
                                               add_slope = False,
                                               epsilon = 0.0)
        if config.getboolean("output_options","output_course_filled_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_filled_orog.nc"),
                                             orography_30min,
                                             fieldname="z")#config.get("output_fieldname_options",
                                                        #          "output_30min_filled_orog"))
        #Transform any necessary field into the necessary format and save ready for parameter generation
        if print_timing_info:
            time_before_parameter_generation = timer()
        transformed_course_rdirs_filename = path.join(self.working_directory_path,"30minute_river_dirs_temp.nc")
        transformed_HD_filled_orography_filename = path.join(self.working_directory_path,"30minute_filled_orog_temp.nc")
        transformed_HD_ls_mask_filename = path.join(self.working_directory_path,"30minute_ls_mask_temp.nc")
        half_degree_grid_filepath = path.join(self.ancillary_data_path,"grid_0_5.txt")
        #rdirs_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        iodriver.advanced_field_writer(transformed_course_rdirs_filename,
                                         rdirs_30min,
                                         fieldname="rdirs")#config.get("output_fieldname_options",
                                                   #           "output_30min_rdirs"))
        #orography_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        iodriver.advanced_field_writer(transformed_HD_filled_orography_filename,
                                         orography_30min,
                                         fieldname="z")#config.get("output_fieldname_options",
                                                   #           "output_30min_orography"))
        #ls_mask_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        ls_mask_30min.invert_data()
        iodriver.advanced_field_writer(transformed_HD_ls_mask_filename,
                                       ls_mask_30min,
                                       fieldname="lsm")#config.get("output_fieldname_options",
                                                       #     "output_30min_ls_mask"))
        #If required generate flow parameters and create a hdparas.file otherwise
        #use a river direction file with coordinates as the hdparas.file
        if config.getboolean("general_options","generate_flow_parameters"):
            #Generate parameters
            self._generate_flow_parameters(rdir_file=transformed_course_rdirs_filename,
                                           topography_file=transformed_HD_filled_orography_filename,
                                           inner_slope_file=\
                                           path.join(self.ancillary_data_path,'bin_innerslope.dat'),
                                           lsmask_file=transformed_HD_ls_mask_filename,
                                           null_file=\
                                           path.join(self.ancillary_data_path,'null.dat'),
                                           area_spacing_file=\
                                           path.join(self.ancillary_data_path,
                                                     'fl_dp_dl.dat'),
                                           orography_variance_file=\
                                           path.join(self.ancillary_data_path,'bin_toposig.dat'),
                                           output_dir=path.join(self.working_directory_path,'paragen'),
                                           production_run=True)
            #Place parameters and rdirs into a hdparas.file
            self._generate_hd_file(rdir_file=path.splitext(transformed_course_rdirs_filename)[0] + ".dat",
                                   lsmask_file=path.splitext(transformed_HD_ls_mask_filename)[0] + ".dat",
                                   null_file=\
                                   path.join(self.ancillary_data_path,'null.dat'),
                                   area_spacing_file=\
                                   path.join(self.ancillary_data_path,
                                             'fl_dp_dl.dat'),
                                   hd_grid_specs_file=half_degree_grid_filepath,
                                   output_file=self.output_hdparas_filepath,
                                   paras_dir=path.join(self.working_directory_path,'paragen'),
                                   production_run=True)
        else:
            #Use a river direction file including coordinates as the hdparas file
            shutil.copy2(transformed_course_rdirs_filename,self.output_hdparas_filepath)
        if self.output_hdstart_filepath is not None:
            utilities.prepare_hdrestart_file_driver(base_hdrestart_filename=base_hd_restart_file,
                                                    output_hdrestart_filename=\
                                                    self.output_hdstart_filepath,
                                                    hdparas_filename=self.output_hdparas_filepath,
                                                    ref_hdparas_filename=ref_hd_paras_file,
                                                    timeslice=None,
                                                    res_num_data_rotate180lr=False,
                                                    res_num_data_flipup=False,
                                                    res_num_ref_rotate180lr=False,
                                                    res_num_ref_flipud=False, grid_type='HD')
        #Post processing
        if print_timing_info:
            time_before_30min_post_processing_two = timer()
        flowtocell_30min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_30min.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat30,
                                                                                    nlong=nlong30),
                                                                                    grid='HD')
        catchments_30min = comp_catchs.compute_catchments_cpp(rdirs_30min.get_data(),
                                                              loops_log_30min_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_30min_filename),
                                       grid="HD")
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic','HD')
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic','HD')
        if config.getboolean("output_options","output_course_flowtocell"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_flowtocell.nc"),
                                           flowtocell_30min,
                                           fieldname="acc")#config.get("output_fieldname_options",
                                                      #          "output_30min_flow_to_cell"))
        if config.getboolean("output_options","output_course_flowtorivermouths"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "30min_flowtorivermouths.nc"),
                                           flowtorivermouths_30min,
                                           fieldname="accrm")#config.get("output_fieldname_options",
                                                     #           "output_30min_flow_to_river_mouths"))
        if config.getboolean("output_options","output_course_catchments"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_catchments.nc"),
                                           catchments_30min,
                                           fieldname="catch")#config.get("output_fieldname_options",
                                                             #   "output_30min_catchments"))
        minima_filename = path.join(self.working_directory_path,
                                    "minima_temp.nc")
        temp_10min_orog_filename = path.join(self.working_directory_path,
                                             "10min_orog_temp.nc")
        cell_areas_filename_10min = path.join(self.ancillary_data_path,
                                              "10min_grid_area_default_R.nc")
        temp_10min_rdirs_filename = path.join(self.working_directory_path,
                                              "10min_rdirs_temp.nc")
        temp_30min_rdirs_filename = path.join(self.working_directory_path,
                                              "30min_rdirs_temp.nc")
        temp_10min_catchments_filename = path.join(self.working_directory_path,
                                                   "10min_catchments_temp.nc")
        temp_30min_catchments_filename = path.join(self.working_directory_path,
                                                   "30min_catchments_temp.nc")
        temp_basin_catchment_numbers_filename = path.join(self.working_directory_path,
                                                          "basin_catchment_numbers_temp.nc")
        output_lakestart_dirname = path.dirname(self.output_lakestart_filepath)
        output_water_redistributed_to_lakes_file = path.join(output_lakestart_dirname,
                                                             "water_to_lakes.nc")
        output_water_redistributed_to_rivers_file = path.join(output_lakestart_dirname,
                                                             "water_to_rivers.nc")
        iodriver.advanced_field_writer(temp_10min_orog_filename,orography_10min,
                                       fieldname="Topo")
        iodriver.advanced_field_writer(temp_10min_catchments_filename,catchments_10min,
                                       fieldname="catchments")
        iodriver.advanced_field_writer(temp_30min_catchments_filename,catchments_30min,
                                       fieldname="catchments")
        iodriver.advanced_field_writer(temp_10min_rdirs_filename,rdirs_10min,
                                       "FDIR")
        iodriver.advanced_field_writer(temp_30min_rdirs_filename,rdirs_30min,
                                       "FDIR")
        #Generate Minima
        utilities.advanced_extract_true_sinks_from_rdirs(rdirs_filename=
                                                         temp_10min_rdirs_filename,
                                                         truesinks_filename=
                                                         minima_filename,
                                                         rdirs_fieldname="FDIR",
                                                         truesinks_fieldname="minima")
        dynamic_lake_operators.\
          advanced_basin_evaluation_driver(input_minima_file=
                                           minima_filename,
                                           input_minima_fieldname="minima",
                                           input_raw_orography_file=temp_10min_orog_filename,
                                           input_raw_orography_fieldname="Topo",
                                           input_corrected_orography_file=temp_10min_orog_filename,
                                           input_corrected_orography_fieldname="Topo",
                                           input_cell_areas_file= cell_areas_filename_10min,
                                           input_cell_areas_fieldname="cell_area",
                                           input_prior_fine_rdirs_file=temp_10min_rdirs_filename,
                                           input_prior_fine_rdirs_fieldname="FDIR",
                                           input_prior_fine_catchments_file=
                                           temp_10min_catchments_filename,
                                           input_prior_fine_catchments_fieldname="catchments",
                                           input_coarse_catchment_nums_file=
                                           temp_30min_catchments_filename,
                                           input_coarse_catchment_nums_fieldname="catchments",
                                           input_coarse_rdirs_file=
                                           temp_30min_rdirs_filename,
                                           input_coarse_rdirs_fieldname="FDIR",
                                           combined_output_filename=
                                           self.output_lakeparas_filepath,
                                           output_filepath=self.working_directory_path,
                                           output_filelabel="temp",
                                           output_basin_catchment_nums_filepath=
                                           temp_basin_catchment_numbers_filename)
        dynamic_lake_operators.\
            advanced_water_redistribution_driver(input_lake_numbers_file=
                                                 temp_basin_catchment_numbers_filename,
                                                 input_lake_numbers_fieldname=
                                                 "basin_catchment_numbers",
                                                 input_lake_centers_file=
                                                 minima_filename,
                                                 input_lake_centers_fieldname=
                                                 "minima",
                                                 input_water_to_redistribute_file=
                                                 self.input_water_to_redistribute_filepath,
                                                 input_water_to_redistribute_fieldname=
                                                 "lake_field",
                                                 output_water_redistributed_to_lakes_file=
                                                 output_water_redistributed_to_lakes_file,
                                                 output_water_redistributed_to_lakes_fieldname=
                                                 "water_redistributed_to_lakes",
                                                 output_water_redistributed_to_rivers_file=
                                                 output_water_redistributed_to_rivers_file,
                                                 output_water_redistributed_to_rivers_fieldname=
                                                 "water_redistributed_to_rivers",
                                                 coarse_grid_type="HD")
        cdo_inst = cdo.Cdo()
        cdo_inst.merge(input=" ".join([output_water_redistributed_to_lakes_file,
                                       output_water_redistributed_to_rivers_file]),
                       output=self.output_lakestart_filepath)
        os.remove(output_water_redistributed_to_lakes_file)
        os.remove(output_water_redistributed_to_rivers_file)
        os.remove(temp_10min_orog_filename)
        os.remove(temp_10min_rdirs_filename)
        os.remove(temp_30min_rdirs_filename)
        os.remove(temp_10min_catchments_filename)
        os.remove(temp_30min_catchments_filename)
        os.remove(minima_filename)
        # if print_timing_info:
        #     end_time = timer()
        #     print "---- Timing info ----"
        #     print "Initial setup:        {: 6.2f}s".\
        #         format(time_before_sink_filling - start_time)
        #     print "River Carving:        {: 6.2f}s".\
        #         format(time_before_10min_post_processing - time_before_sink_filling)
        #     print "Post Processing:      {: 6.2f}s".\
        #         format(time_before_upscaling - time_before_10min_post_processing)
        #     print "Upscaling:            {: 6.2f}s".\
        #         format(time_before_30min_post_processing_one - time_before_upscaling)
        #     print "Post Processing:      {: 6.2f}s".\
        #         format(time_before_loop_breaker -
        #                 time_before_30min_post_processing_one)
        #     print "Loop Breaker:         {: 6.2f}s".\
        #         format(time_before_sink_filling - time_before_loop_breaker)
        #     print "Sink Filling:         {: 6.2f}s".\
        #         format(time_before_parameter_generation - time_before_sink_filling)
        #     print "Parameter Generation: {: 6.2f}s".\
        #         format(time_before_30min_post_processing_two -
        #                 time_before_parameter_generation)
        #     print "Post Processing:      {: 6.2f}s".\
        #         format(end_time - time_before_30min_post_processing_two)
        #     print "Total:                {: 6.2f}s".\
        #         format(end_time-start_time)

def setup_and_run_dynamic_hd_para_and_lake_gen_from_command_line_arguments(args):
    """Setup and run a dynamic hd production run from the command line arguments passed in by main"""
    driver_object = Dynamic_Lake_Production_Run_Drivers(**vars(args))
    driver_object.no_intermediaries_dynamic_lake_driver()

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
    parser.add_argument('input_orography_filepath',
                        metavar='input-orography-filepath',
                        help='Full path to input orography to use',
                        type=str)
    parser.add_argument('input_ls_mask_filepath',
                        metavar='input-ls-mask-filepath',
                        help='Full path to input land sea mask to use',
                        type=str)
    parser.add_argument('input_water_to_redistribute_filepath',
                        metavar='input-water-to-redistribute-filepath',
                        help='Full path to file containing water from lakes in previous run',
                        type=str)
    parser.add_argument('present_day_base_orography_filepath',
                        metavar='present-day-base-orography-filepath',
                        help='Full path to present day orography input orography is based on',
                        type=str)
    parser.add_argument('glacier_mask_filepath',
                        metavar='glacier-mask-filepath',
                        help='Full path to input glacier mask file',
                        type=str)
    parser.add_argument('output_hdparas_filepath',
                        metavar='output-hdparas-filepath',
                        help='Full path to target destination for output hdparas file',
                        type=str)
    parser.add_argument('output_lakeparas_filepath',
                        metavar='output-lakeparas-filepath',
                        help='Full path to target destination for output lakeparas file',
                        type=str)
    parser.add_argument('ancillary_data_directory',
                        metavar='ancillary-data-directory',
                        help='Full path to directory containing ancillary data',
                        type=str)
    parser.add_argument('working_directory',
                        metavar='working-directory',
                        help='Full path to target working directory',
                        type=str)
    parser.add_argument('output_lakestart_filepath',
                        metavar='output-lakestart-filepath',
                        help='Full path to target destination for output lakestart file ',
                        type=str)
    parser.add_argument('-s','--output-hdstart-filepath',
                        help='Full path to target destination for output hdstart file',
                        type=str,
                        default=None)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_dynamic_hd_para_and_lake_gen_from_command_line_arguments(args)
