'''
Created on April 1, 2020

@author: thomasriddick
'''
import determine_river_directions
import dynamic_hd_driver as dyn_hd_dr

class Dynamic_Lake_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):
    """A class with methods used for running a production run of the dynamic HD and Lake generation code"""

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
                                                   "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                   "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                   "_glcc_olson_lsmask_0k_20170517_003802.nc")
        #Change ls mask to correct type
        ls_mask_10min_fieldname = config.get("input_fieldname_options",
                                             "input_landsea_mask_fieldname")
        ls_mask_10min_fieldname = ls_mask_10min_fieldname if ls_mask_10min_fieldname else None
        ls_mask_10min = dynamic_hd.load_field(self.original_ls_mask_filename,
                                              file_type=\
                                              dynamic_hd.get_file_extension(self.original_ls_mask_filename),
                                              field_type='Generic',
                                              fieldname=ls_mask_10min_fieldname,
                                              unmask=False,
                                              timeslice=None,
                                              grid_type='LatLong10min')
        ls_mask_10min.change_dtype(np.int32)
        #Add corrections to orography
        orography_10min_fieldname = config.get("input_fieldname_options",
                                               "input_orography_fieldname")
        orography_10min_fieldname = orography_10min_fieldname if orography_10min_fieldname else None
        orography_10min = dynamic_hd.load_field(self.original_orography_filename,
                                                file_type=dynamic_hd.\
                                                get_file_extension(self.original_orography_filename),
                                                fieldname=orography_10min_fieldname,
                                                field_type='Orography', grid_type="LatLong10min")
        if self.present_day_base_orography_filename:
            present_day_base_orography_fieldname = config.get("input_fieldname_options",
                                                              "input_base_present_day_orography_fieldname")
            present_day_base_orography_fieldname = present_day_base_orography_fieldname if \
                                                   present_day_base_orography_fieldname else None
            present_day_base_orography = \
            dynamic_hd.load_field(self.present_day_base_orography_filename,
                                  file_type=dynamic_hd.\
                                  get_file_extension(self.present_day_base_orography_filename),
                                  field_type='Orography',
                                  fieldname=present_day_base_orography_fieldname,
                                  grid_type="LatLong10min")
            present_day_reference_orography = \
            dynamic_hd.load_field(present_day_reference_orography_filename,
                                  file_type=dynamic_hd.\
                                  get_file_extension(present_day_reference_orography_filename),
                                  field_type='Orography',
                                  grid_type="LatLong10min")
            orography_10min = utilities.rebase_orography(orography=orography_10min,
                                                         present_day_base_orography=\
                                                         present_day_base_orography,
                                                         present_day_reference_orography=\
                                                         present_day_reference_orography)
        orography_corrections_10min =  dynamic_hd.load_field(orography_corrections_filename,
                                                             file_type=dynamic_hd.\
                                                             get_file_extension(orography_corrections_filename),
                                                             field_type='Orography', grid_type="LatLong10min")
        orography_uncorrected_10min = orography_10min.copy()
        orography_10min.add(orography_corrections_10min)
        if self.glacier_mask_filename:
            glacier_mask_10min_fieldname = config.get("input_fieldname_options",
                                                "input_glacier_mask_fieldname")
            glacier_mask_10min_fieldname = glacier_mask_10min_fieldname if glacier_mask_10min_fieldname else 'sftgif'
            glacier_mask_10min = dynamic_hd.load_field(self.glacier_mask_filename,
                                                       file_type=dynamic_hd.\
                                                       get_file_extension(self.glacier_mask_filename),
                                                       fieldname=glacier_mask_10min_fieldname,
                                                       field_type='Orography',
                                                       unmask=True,grid_type="LatLong10min")
            orography_10min = utilities.\
            replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography=\
                                                                                orography_10min,
                                                                                input_original_orography=\
                                                                                orography_uncorrected_10min,
                                                                                input_glacier_mask=
                                                                                glacier_mask_10min)
        if config.getboolean("output_options","output_corrected_orog"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_corrected_orog.nc"),
                                   orography_10min,
                                   file_type=".nc")
        #Generate orography with filled sinks
        if print_timing_info:
            time_before_sink_filling = timer()
        grid_dims_10min=orography_10min.get_grid().get_grid_dimensions()
        truesinks = Field(np.empty((1,1),dtype=np.int32),grid='HD')
        ls_mask_10min.flip_data_ud()
        orography_10min.flip_data_ud()
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
        if config.getboolean("output_options","output_fine_rdirs"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_rdirs.nc"),
                                   rdirs_10min,
                                   file_type=".nc")

        #Filter unfilled orography
        lake_operators_wrapper.filter_out_shallow_lakes(orography_10min.get_data(),
                                                        np.ascontiguousarray(orography_10min_filled.\
                                                                             get_data(),
                                                                             dtype=np.float64),
                                                        minimum_depth_threshold=5.0)
        #Generate River Directions
        determine_river_directions.determine_river_directions(orography=orography_10min,
                                                              lsmask=ls_mask_10min,
                                                              truesinks=None,
                                                              always_flow_to_sea=True,
                                                              use_diagonal_nbrs=True,
                                                              mark_pits_as_true_sinks=True)
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
        catchments_10min = comp_catchs.compute_catchments_cpp(rdirs_10min.get_data(),
                                                              loops_log_filename)
        catchments_10min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_10min,
                                                                               loops_log_filename),
                                       grid="HD")
        if config.getboolean("output_options","output_fine_flowtocell"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_flowtocell.nc"),
                                   flowtocell_10min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_fine_flowtorivermouths"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_flowtorivermouths.nc"),
                                   flowtorivermouths_10min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_fine_catchments"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_catchments.nc"),
                                   catchment_10min,
                                   file_type=".nc")
        #Run Upscaling
        if print_timing_info:
            time_before_upscaling = timer()
        loops_log_filename = path.join(self.working_directory_path,"loops.log")
        catchments_log_filename= path.join(self.working_directory_path,"catchments.log")
        cotat_plus_parameters_filename = path.join(self.ancillary_data_path,'cotat_plus_standard_params.nl')
        rdirs_30min = run_cotat_plus(rdirs_10min, flowtocell_10min,
                                      cotat_plus_parameters_filename,'HD')
        if config.getboolean("output_options","output_pre_loop_removal_course_rdirs"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_rdirs.nc"),
                                   rdirs_30min,
                                   file_type=".nc")
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
                                                              loops_log_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_filename),
                                       grid="HD")
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic','HD')
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic','HD')
        if config.getboolean("output_options","output_pre_loop_removal_course_flowtocell"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_flowtocell.nc"),
                                   flowtocell_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_pre_loop_removal_course_flowtorivermouths"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_flowtorivermouths.nc"),
                                   flowtorivermouths_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_pre_loop_removal_course_catchments"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_catchments.nc"),
                                   catchments_30min,
                                   file_type=".nc")
        #Run Loop Breaker
        if print_timing_info:
            time_before_loop_breaker = timer()
        loop_nums_list = []
        first_line_pattern = re.compile(r"^Loops found in catchments:$")
        with open(loops_log_filename,'r') as f:
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
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_rdirs.nc"),
                                   rdirs_30min,
                                   file_type=".nc")
        #Upscale the orography to the HD grid for calculating the flow parameters
        orography_30min= utilities.upscale_field(orography_10min,"HD","Sum",{},
                                                 scalenumbers=True)
        if config.getboolean("output_options","output_course_unfilled_orog"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_unfilled_orog.nc"),
                                   orography_30min,
                                   file_type=".nc")
        #Extract HD ls mask from river directions
        ls_mask_30min = field.RiverDirections(rdirs_30min.get_lsmask(),grid='HD')
        #Fill HD orography for parameter generation
        if print_timing_info:
            time_before_sink_filling = timer()
        truesinks = Field(np.empty((1,1),dtype=np.int32),grid='HD')
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
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_filled_orog.nc"),
                                   orography_30min,
                                   file_type=".nc")
        #Transform any necessary field into the necessary format and save ready for parameter generation
        if print_timing_info:
            time_before_parameter_generation = timer()
        transformed_course_rdirs_filename = path.join(self.working_directory_path,"30minute_river_dirs_temp.nc")
        transformed_HD_filled_orography_filename = path.join(self.working_directory_path,"30minute_filled_orog_temp.nc")
        transformed_HD_ls_mask_filename = path.join(self.working_directory_path,"30minute_ls_mask_temp.nc")
        half_degree_grid_filepath = path.join(self.ancillary_data_path,"grid_0_5.txt")
        rdirs_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        dynamic_hd.write_field(filename=transformed_course_rdirs_filename,
                               field=rdirs_30min,
                               file_type=\
                               dynamic_hd.get_file_extension(transformed_course_rdirs_filename),
                               griddescfile=half_degree_grid_filepath)
        orography_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        dynamic_hd.write_field(filename=transformed_HD_filled_orography_filename,
                               field=orography_30min,
                               file_type=dynamic_hd.\
                               get_file_extension(transformed_HD_filled_orography_filename),
                               griddescfile=half_degree_grid_filepath)
        ls_mask_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        ls_mask_30min.invert_data()
        dynamic_hd.write_field(filename=transformed_HD_ls_mask_filename,
                               field=ls_mask_30min,
                               file_type=dynamic_hd.\
                               get_file_extension(transformed_HD_ls_mask_filename),
                               griddescfile=half_degree_grid_filepath)
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
                                                              loops_log_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_filename),
                                       grid="HD")
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic','HD')
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic','HD')
        if config.getboolean("output_options","output_course_flowtocell"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_flowtocell.nc"),
                                   flowtocell_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_course_flowtorivermouths"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_flowtorivermouths.nc"),
                                   flowtorivermouths_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_course_catchments"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_catchments.nc"),
                                   catchments_30min,
                                   file_type=".nc")
        #Generate Minima
        working_orog_icemask.invert_data()
        if (invert_ls_mask):
          working_orog_lsmask.invert_data()
        if (rotate_lsmask_180_lr):
          working_orog_lsmask.rotate_field_by_a_hundred_and_eighty_degrees()
        working_orog_field.mask_field_with_external_mask(working_orog_icemask.get_data())
        working_orog_sinkless_field.update_field_with_partially_masked_data(working_orog_field)
        working_orog_field.mask_field_with_external_mask(working_orog_lsmask.get_data())
        working_orog_sinkless_field.update_field_with_partially_masked_data(working_orog_field)
        iodriver.advanced_field_writer(output_working_orog_sinkless_improved_filename,
                                       working_orog_sinkless_field,fieldname="Topo",clobber=True)
        dynamic_lake_operators.advanced_local_minima_finding_driver(output_filtered_working_orog_filename,
                                                                    "Topo",
                                                                    minima_working_orog_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_working_orog_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename_working_orog,
                                                                minima_fieldname)
                minima_from_rdirs_filename = ("/Users/thomasriddick/Documents/data/HDdata/minima/"
                                      "minima_" + file_label + "_reduced"
                                      "_" + str(timestep) + "_landonly_from_rdirs.nc")
        utilities.advanced_extract_true_sinks_from_rdirs(rdirs_filename=
                                                         "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                                         "updated_RFDs_" + file_label + "_10min_with_depressions.nc",
                                                         truesinks_filename=
                                                         minima_from_rdirs_filename,
                                                         rdirs_fieldname="FDIR",
                                                         truesinks_fieldname="minima")
        dynamic_lake_operators.\
          advanced_basin_evaluation_driver(input_minima_file=
                                           minima_from_rdirs_filename,
                                           input_minima_fieldname="minima",
                                           input_raw_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_" + str(timestep) +
                                           "_ice6g_lake_filtered_" + file_label + ".nc",
                                           input_raw_orography_fieldname="Topo",
                                           input_corrected_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_" + str(timestep) +
                                           "_ice6g_lake_filtered_" + file_label + ".nc",
                                           input_corrected_orography_fieldname="Topo",
                                           input_cell_areas_file= cell_areas_filename_10min,
                                           input_cell_areas_fieldname="cell_area",
                                           input_prior_fine_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_" + file_label + "_10min_with_depressions.nc",
                                           input_prior_fine_rdirs_fieldname="FDIR",
                                           input_prior_fine_catchments_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_" + file_label + "_10mins.nc",
                                           input_prior_fine_catchments_fieldname="catchments",
                                           input_coarse_catchment_nums_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_" + file_label + "_30mins.nc",
                                           input_coarse_catchment_nums_fieldname="catchments",
                                           input_coarse_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_" + file_label + "_30min_with_depressions.nc",
                                           input_coarse_rdirs_fieldname="FDIR",
                                           combined_output_filename=
                                            join(self.lake_parameter_file_path,
                                                 "lakeparas_" + file_label + ".nc"),
                                           output_filepath=self.lake_parameter_file_path,
                                           output_filelabel=file_label,
                                           output_basin_catchment_nums_filepath=
                                           join(self.basin_catchment_numbers_path,
                                                "basin_catchment_numbers_" + file_label + ".nc"))
        if print_timing_info:
            end_time = timer()
            print "---- Timing info ----"
            print "Initial setup:        {: 6.2f}s".\
                format(time_before_sink_filling - start_time)
            print "River Carving:        {: 6.2f}s".\
                format(time_before_10min_post_processing - time_before_sink_filling)
            print "Post Processing:      {: 6.2f}s".\
                format(time_before_upscaling - time_before_10min_post_processing)
            print "Upscaling:            {: 6.2f}s".\
                format(time_before_30min_post_processing_one - time_before_upscaling)
            print "Post Processing:      {: 6.2f}s".\
                format(time_before_loop_breaker -
                        time_before_30min_post_processing_one)
            print "Loop Breaker:         {: 6.2f}s".\
                format(time_before_sink_filling - time_before_loop_breaker)
            print "Sink Filling:         {: 6.2f}s".\
                format(time_before_parameter_generation - time_before_sink_filling)
            print "Parameter Generation: {: 6.2f}s".\
                format(time_before_30min_post_processing_two -
                        time_before_parameter_generation)
            print "Post Processing:      {: 6.2f}s".\
                format(end_time - time_before_30min_post_processing_two)
            print "Total:                {: 6.2f}s".\
                format(end_time-start_time)

def setup_and_run_dynamic_hd_para_gen_from_command_line_arguments(args):
    """Setup and run a dynamic hd production run from the command line arguments passed in by main"""
    driver_object = Dynamic_HD_Production_Run_Drivers(**vars(args))
    driver_object.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()

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
    parser.add_argument('ancillary_data_directory',
                        metavar='ancillary-data-directory',
                        help='Full path to directory containing ancillary data',
                        type=str)
    parser.add_argument('working_directory',
                        metavar='working-directory',
                        help='Full path to target working directory',
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
    setup_and_run_dynamic_hd_para_gen_from_command_line_arguments(args)
