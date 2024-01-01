import re
import argparse
import shutil
import os
import warnings
import pathlib
from os.path import join
from datetime import datetime
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_hd_driver \
    import Dynamic_HD_Drivers
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_lake_production_run_driver \
    import Dynamic_Lake_Production_Run_Drivers
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_hd_production_run_driver \
    import Dynamic_HD_Production_Run_Drivers
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_lake_corrections_production_driver \
    import Dynamic_Lake_Correction_Production_Run_Drivers
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_lake_corrections_production_driver \
    import CorrectionTypes

def replace_rightmost(s,oldvalue,newvalue):
    string_as_list = s.rsplit(oldvalue,1)
    return newvalue.join(string_as_list)

class Dynamic_Lake_Analysis_Run_Framework:

    ancillary_data_directory_match = re.compile(r"ancillary data directory:\s*(\S*)")
    present_day_base_orography_filepath_match = re.compile(r"present day base orography file:\s*(\S*)")
    base_corrections_filepath_match = re.compile(r"base corrections file:\s*(\S*)")
    base_date_based_corrections_filepath_match = re.compile(r"base date-based corrections file:\s*(\S*)")
    base_additional_corrections_filepath_match = re.compile(r"base additional corrections file:\s*(\S*)")
    base_true_sinks_filepath_match = re.compile(r"base true sinks file:\s*(\S*)")
    orography_filepath_template_match = re.compile(r"orography file template:\s*(\S*)")
    landsea_mask_filepath_template_match = re.compile(r"landsea mask file template:\s*(\S*)")
    glacier_mask_filepath_template_match = re.compile(r"glacier mask file template:\s*(\S*)")
    version_match = re.compile(r"Version\s*(\d+)\s*-")

    def __init__(self,
                 base_directory,
                 setup_directory_structure=False,
                 ancillary_data_directory=None,
                 present_day_base_orography_filepath=None,
                 base_corrections_filepath=None,
                 base_date_based_correction_filepath=None,
                 base_additional_corrections_filepath=None,
                 base_true_sinks_filepath=None,
                 orography_filepath_template=None,
                 landsea_mask_filepath_template=None,
                 glacier_mask_filepath_template=None,
                 generate_lake_orography_corrections=False,
                 apply_orography_tweaks=False,
                 change_date_based_corrections=False,
                 make_analysis_run=False,
                 skip_dynamic_river_production=False,
                 skip_dynamic_lake_production=False,
                 skip_current_day_time_slice=False,
                 run_hd_scripting_default_orography_corrections=False,
                 start_date=0,end_date=0,
                 slice_spacing=10,
                 clear_lake_results=False,
                 clear_river_results=False,
                 clear_river_default_orog_corrs_results=False,
                 generate_present_day_rivers_with_original_sink_set=False,
                 generate_present_day_rivers_with_true_sinks=False):
        self.base_directory = base_directory
        self.lakes_directory = join(base_directory,"lakes")
        self.rivers_directory = join(base_directory,"rivers")
        self.corrections_directory = join(base_directory,"corrections")
        self.setup_directory_structure = setup_directory_structure
        if ancillary_data_directory is not None:
            self.ancillary_data_directory = ancillary_data_directory
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.ancillary_data_directory_match,
                                  ancillary_data_directory,
                                  self.base_directory)
        else:
            self.ancillary_data_directory = self.read_info(self.ancillary_data_directory_match,
                                                           self.base_directory)
        if present_day_base_orography_filepath is not None:
            self.present_day_base_orography_filepath = present_day_base_orography_filepath
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.present_day_base_orography_filepath_match,
                                  present_day_base_orography_filepath,
                                  self.base_directory)
        else:
            self.present_day_base_orography_filepath =\
                self.read_info(self.present_day_base_orography_filepath_match,
                               self.base_directory)
        if base_corrections_filepath is not None:
            self.base_corrections_filepath = base_corrections_filepath
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.base_corrections_filepath_match,
                                  base_corrections_filepath,
                                  self.base_directory)
        else:
            self.base_corrections_filepath =\
                self.read_info(self.base_corrections_filepath_match,
                               self.base_directory)
        if base_date_based_correction_filepath is not None:
            self.base_date_based_correction_filepath = base_date_based_correction_filepath
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.base_date_based_corrections_filepath_match,
                                  base_date_based_correction_filepath,
                                  self.base_directory)
        else:
            self.base_date_based_correction_filepath =\
                self.read_info(self.base_date_based_corrections_filepath_match,
                               self.base_directory)
        if base_additional_corrections_filepath is not None:
            self.base_additional_corrections_filepath = \
                base_additional_corrections_filepath
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.base_additional_corrections_filepath,
                                  base_additional_corrections_filepath,
                                  self.base_directory)
        else:
            self.base_additional_corrections_filepath =\
                self.read_info(self.base_additional_corrections_filepath_match,
                               self.base_directory)
        if base_true_sinks_filepath is not None:
            self.base_true_sinks_filepath = base_true_sinks_filepath
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.base_true_sinks_filepath_match,
                                  base_true_sinks_filepath,
                                  self.base_directory)
        else:
            self.base_true_sinks_filepath =\
                self.read_info(self.base_true_sinks_filepath_match,
                               self.base_directory)
        if orography_filepath_template is not None:
            self.orography_filepath_template = orography_filepath_template
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.orography_filepath_template_match,
                                  orography_filepath_template,
                                  self.base_directory)
        else:
            self.orography_filepath_template = \
                self.read_info(self.orography_filepath_template_match,
                               self.base_directory)
        if landsea_mask_filepath_template is not None:
            self.landsea_mask_filepath_template = landsea_mask_filepath_template
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.landsea_mask_filepath_template_match,
                                  landsea_mask_filepath_template,
                                  self.base_directory)
        else:
            self.landsea_mask_filepath_template = \
                self.read_info(self.landsea_mask_filepath_template_match,
                               self.base_directory)
        if glacier_mask_filepath_template is not None:
            self.glacier_mask_filepath_template = glacier_mask_filepath_template
            if os.path.isfile(join(base_directory,"analysis_info.txt")):
                self.replace_info(self.glacier_mask_filepath_template_match,
                                  glacier_mask_filepath_template,
                                  self.base_directory)
        else:
            self.glacier_mask_filepath_template = \
                self.read_info(self.glacier_mask_filepath_template_match,
                               self.base_directory)
        self.generate_lake_orography_corrections = generate_lake_orography_corrections
        self.apply_orography_tweaks = apply_orography_tweaks
        self.change_date_based_corrections = change_date_based_corrections
        self.make_analysis_run = make_analysis_run
        self.run_river_scripting = not skip_dynamic_river_production
        self.run_lake_scripting = not skip_dynamic_lake_production
        self.skip_current_day_time_slice = skip_current_day_time_slice
        self.run_hd_scripting_default_orography_corrections = \
            run_hd_scripting_default_orography_corrections
        self.start_date = start_date
        self.end_date = end_date
        self.slice_spacing = slice_spacing
        self.run_clear_lake_results = (clear_lake_results >= -1)
        self.run_clear_river_results = (clear_river_results >= -1)
        self.clear_lake_results_version = clear_lake_results
        self.clear_river_results_version = clear_river_results
        self.run_clear_river_default_orog_corrs_results = clear_river_default_orog_corrs_results
        self.generate_present_day_rivers_with_original_sink_set = \
            generate_present_day_rivers_with_original_sink_set
        self.generate_present_day_rivers_with_true_sinks = \
            generate_present_day_rivers_with_true_sinks
        self.dyn_lake_driver = Dynamic_Lake_Production_Run_Drivers(input_orography_filepath=None,
                                                                   input_ls_mask_filepath=None,
                                                                   input_water_to_redistribute_filepath=None,
                                                                   output_hdparas_filepath=None,
                                                                   output_lakeparas_filepath=None,
                                                                   output_lakestart_filepath=None,
                                                                   ancillary_data_directory=
                                                                   self.ancillary_data_directory,
                                                                   working_directory=join(self.base_directory,
                                                                                          "lakes","work"),
                                                                   output_hdstart_filepath=None,
                                                                   present_day_base_orography_filepath=
                                                                   self.present_day_base_orography_filepath,
                                                                   glacier_mask_filepath=None,
                                                                   non_standard_orog_correction_filename=None)
        self.dyn_hd_driver = Dynamic_HD_Production_Run_Drivers(input_orography_filepath=None,
                                                               input_ls_mask_filepath=None,
                                                               output_hdparas_filepath=None,
                                                               ancillary_data_directory=
                                                               self.ancillary_data_directory,
                                                               working_directory=join(self.base_directory,
                                                                                      "rivers","work"),
                                                               output_hdstart_filepath=None,
                                                               present_day_base_orography_filepath=
                                                               self.present_day_base_orography_filepath,
                                                               glacier_mask_filepath=None,
                                                               non_standard_orog_correction_filename=None)
        self.dyn_lake_correction_driver = \
            Dynamic_Lake_Correction_Production_Run_Drivers(working_directory=join(self.base_directory,
                                                                                  "corrections","work"))
        if self.setup_directory_structure:
            self.corrections_version = 0
        else:
            #Version should be in order but don't assume this
            version_temp = 0
            self.corrections_version = 0
            with open(join(base_directory,"versions.txt"),"r") as info_txt:
                for line in info_txt:
                    match = self.version_match.match(line)
                    if match:
                        version_temp = int(match.group(1))
                        if version_temp > self.corrections_version:
                            self.corrections_version = version_temp
            self.corrections_file_for_current_version = join(self.corrections_directory,
                                                             "correction_fields",
                                                             "correction_field_version_{}.nc".\
                                                             format(self.corrections_version))
        if len(os.listdir(join(self.corrections_directory,
                               "date_based_correction_sets"))) != 0:
            self.date_based_corrections_file_for_current_version = join(self.corrections_directory,
                                                                        "date_based_correction_sets",
                                                                        "date_based_corrections_set_version_{}.nc".\
                                                                        format(self.corrections_version))
        else:
            self.date_based_corrections_file_for_current_version = None

        if len(os.listdir(join(self.corrections_directory,
                               "additional_correction_sets"))) != 0:
            self.additional_corrections_file_for_current_version = join(self.corrections_directory,
                                                                        "additional_correction_sets",
                                                                        "additional_corrections_set_version_{}.nc".\
                                                                        format(self.corrections_version))
        else:
            self.additional_corrections_file_for_current_version = None

    @staticmethod
    def read_info(required_match,base_directory):
        with open(join(base_directory,"analysis_info.txt"),"r") as info_txt:
            duplicate = False
            for line in info_txt:
                match = required_match.match(line)
                if match:
                    if duplicate:
                        raise RuntimeError("Corrupt analysis_info.txt file")
                    duplicate = True
                    value_read = match.group(1)
        if value_read.lower() == "none":
            value_read = None
        return value_read

    @staticmethod
    def replace_info(required_match,new_value,base_directory):
        try:
            with open(join(base_directory,"analysis_info.txt"),"r") as info_txt:
                with open(join(base_directory,"new_analysis_info.txt"),"w") as new_info_txt:
                    duplicate = False
                    for line in info_txt:
                        match = required_match.match(line)
                        if match:
                            if duplicate:
                                raise RuntimeError("Corrupt analysis_info.txt file")
                            duplicate = True
                            new_line = re.sub(r"(.*:)\s\S*$","\g<1> {}".format(new_value),
                                              line,count=1)
                            new_info_txt.write(new_line)
                        else:
                            new_info_txt.write(line)
            shutil.copyfile(join(base_directory,"new_analysis_info.txt"),
                            join(base_directory,"analysis_info.txt"))
        finally:
            os.remove(join(base_directory,"new_analysis_info.txt"))

    def run_selected_processes(self):
        if self.setup_directory_structure:
            self.run_setup_directory_structure()
        if self.run_clear_lake_results:
            self.clear_lake_results(self.clear_lake_results_version)
        if self.run_clear_river_results:
            self.clear_river_results(self.clear_river_results_version)
        if self.run_clear_river_default_orog_corrs_results:
            self.clear_river_default_orog_corrs_results()
        if (self.generate_lake_orography_corrections or
            self.apply_orography_tweaks or self.change_date_based_corrections):
            self.run_corrections_generation()

        if self.make_analysis_run:
            self.driver_transient_run(start_date=self.start_date,
                                      end_date=self.end_date,
                                      slice_spacing = self.slice_spacing,
                                      dates_to_run_in=None)

    def run_corrections_generation(self):
        if (not self.generate_lake_orography_corrections and
            not self.apply_orography_tweaks and
            not self.change_date_based_corrections):
            warnings.warn("Running correction generation with invalid flags; "
                          "function will have no effect")
            return
        self.corrections_version += 1
        self.corrections_file_for_current_version = join(self.corrections_directory,
                                                         "correction_fields",
                                                         "correction_field_version_{}.nc".\
                                                         format(self.corrections_version))
        if os.path.exists(join(self.corrections_directory,"correction_sets",
                               "corrections_set_version_{}.txt".\
                               format(self.corrections_version))):
            raise RuntimeError("Versioning error - next version number already exists!")
        with open(join(self.corrections_directory,
                       "working_correction_set.txt"),"r") as corrs_txt:
            corrs_txt_data = corrs_txt.read()
        with open(join(self.corrections_directory,"correction_sets",
                       "corrections_set_version_{}.txt".\
                       format(self.corrections_version)),"w") as corrs_txt_copy:
            corrs_txt_copy.write("# Version {} - {}\n".format(self.corrections_version,
                                                              datetime.now()))
            corrs_txt_copy.write(corrs_txt_data)
        with open(join(self.corrections_directory,
                       "working_date_based_correction_set.txt"),"r") as date_based_corrs_txt:
            date_based_corrs_txt_data = date_based_corrs_txt.read()
        with open(join(self.corrections_directory,"date_based_correction_sets",
                       "date_based_corrections_set_version_{}.txt".\
                       format(self.corrections_version)),"w") as date_based_corrs_txt_copy:
            date_based_corrs_txt_copy.write("# Version {} - {}\n".format(self.corrections_version,
                                                                         datetime.now()))
            date_based_corrs_txt_copy.write(date_based_corrs_txt_data)
        with open(join(self.corrections_directory,
                       "working_additional_correction_set.txt"),"r") as additional_corrs_txt:
            additional_corrs_txt_data = additional_corrs_txt.read()
        with open(join(self.corrections_directory,"additional_correction_sets",
                       "additional_corrections_set_version_{}.txt".\
                       format(self.corrections_version)),"w") as additional_corrs_txt_copy:
            additional_corrs_txt_copy.write("# Version {} - {}\n".format(self.corrections_version,
                                                                         datetime.now()))
            additional_corrs_txt_copy.write(additional_corrs_txt_data)
        with open(join(self.corrections_directory,
                       "working_true_sinks_set.txt"),"r") as true_sinks_txt:
            true_sinks_txt_data = true_sinks_txt.read()
        with open(join(self.corrections_directory,"true_sinks_sets",
                       "true_sinks_set_version_{}.txt".\
                       format(self.corrections_version)),"w") as true_sinks_txt_copy:
            true_sinks_txt_copy.write("# Version {} - {}\n".format(self.corrections_version,
                                                                   datetime.now()))
            true_sinks_txt_copy.write(true_sinks_txt_data)
        with open(join(self.base_directory,"versions.txt"),"a") as versions_txt:
            versions_txt.write("Version {} - {}\n".format(self.corrections_version,
                                                          datetime.now()))
        intermediate_processed_orography_corrections_filename = \
                join(self.corrections_directory,"work",
                     "intermediary_processed_orography_corrections.nc")
        processed_orography_corrections_with_final_adjustments_filename = \
                join(self.corrections_directory,"work",
                     "processed_orography_corrections_with_final_adjustments.nc")
        self.dyn_lake_correction_driver.clean_work_dir(partial_clean=False)
        if self.generate_lake_orography_corrections:
            unprocessed_orography_corrections_with_initial_adjustments_filename = \
                join(self.corrections_directory,"work",
                     "unprocessed_orography_with_initial_adjustments.nc")
            true_sinks_filename = join(self.corrections_directory,"true_sinks_fields",
                                       "true_sinks_field_version_{}.nc".\
                                       format(self.corrections_version))
            if os.path.exists(unprocessed_orography_corrections_with_initial_adjustments_filename):
                os.remove(unprocessed_orography_corrections_with_initial_adjustments_filename)
            if os.path.exists(intermediate_processed_orography_corrections_filename):
                os.remove(intermediate_processed_orography_corrections_filename)
            self.dyn_lake_correction_driver.prepare_true_sinks(self.base_true_sinks_filepath,
                                                               true_sinks_filename,
                                                               join(self.corrections_directory,
                                                                    "working_true_sinks_set.txt"))
            self.dyn_lake_correction_driver.apply_tweaks(self.base_corrections_filepath,
                    unprocessed_orography_corrections_with_initial_adjustments_filename,
                    join(self.corrections_directory,
                        "working_correction_set.txt"),
                    CorrectionTypes.PRELIMINARY)
            self.dyn_lake_correction_driver.\
                no_intermediaries_lake_corrections_driver(version=self.corrections_version,
                    original_orog_corrections_filename=
                    unprocessed_orography_corrections_with_initial_adjustments_filename,
                    new_orography_corrections_filename=
                    intermediate_processed_orography_corrections_filename,
                    true_sinks_filename=true_sinks_filename)
            self.dyn_lake_correction_driver.\
                apply_tweaks(intermediate_processed_orography_corrections_filename,
                             self.corrections_file_for_current_version,
                             join(self.corrections_directory,
                                  "working_correction_set.txt"),
                             CorrectionTypes.FINAL)
            self.dyn_lake_correction_driver.clean_work_dir(partial_clean=True)
        elif self.apply_orography_tweaks:
            self.dyn_lake_correction_driver.apply_tweaks(intermediate_processed_orography_corrections_filename,
                                                         self.corrections_file_for_current_version,
                                                         join(self.corrections_directory,
                                                              "working_correction_set.txt"),
                                                         CorrectionTypes.FINAL)
            self.dyn_lake_correction_driver.clean_work_dir(partial_clean=True)
        elif self.change_date_based_corrections:
            shutil.copyfile(intermediate_processed_orography_corrections_filename,
                            self.corrections_file_for_current_version)

    def run_slice(self,slice_time,force_run_all=False):
        slice_label = "version_{}_date_{}".format(self.corrections_version,
                                                  slice_time)
        slice_time_formatted = str(slice_time) if slice_time >= 1000 else "{:04d}".format(slice_time)
        orography_filepath = replace_rightmost(self.orography_filepath_template,
                                               "DATE",slice_time_formatted)
        landsea_mask_filepath = replace_rightmost(self.landsea_mask_filepath_template,
                                                  "DATE",str(slice_time))
        glacier_mask_filepath = replace_rightmost(self.glacier_mask_filepath_template,
                                                  "DATE",str(slice_time))
        if self.run_lake_scripting or force_run_all:
            self.dyn_lake_driver.original_orography_filename=orography_filepath
            self.dyn_lake_driver.original_ls_mask_filename=landsea_mask_filepath
            #Use a random file for now
            warnings.warn("Using dummy water to redistribute file")
            self.dyn_lake_driver.input_water_to_redistribute_filepath=\
            "/Users/thomasriddick/Documents/data/simulation_data/laketestdata/lake_volumes_pmt0531_Tom_41091231.nc"
            self.dyn_lake_driver.output_hdparas_filepath=\
                join(self.lakes_directory,
                     "results","hdpara_{}.nc".format(slice_label))
            self.dyn_lake_driver.output_lakeparas_filepath=\
                join(self.lakes_directory,
                     "results","lakeparas_{}.nc".format(slice_label))
            self.dyn_lake_driver.output_lakestart_filepath=\
                join(self.lakes_directory,
                     "results","lakestart_{}.nc".format(slice_label))
            self.dyn_lake_driver.output_hdstart_filepath=\
                join(self.lakes_directory,
                     "results","hdstart_{}.nc".format(slice_label))
            self.dyn_lake_driver.glacier_mask_filename=glacier_mask_filepath
            self.dyn_lake_driver.non_standard_orog_correction_filename=\
                (self.base_corrections_filepath if self.corrections_version == 0 else
                 self.corrections_file_for_current_version)
            self.dyn_lake_driver.date_based_sill_height_corrections_list_filename = \
                self.date_based_corrections_file_for_current_version
            self.dyn_lake_driver.additional_orography_corrections_list_filepath = \
                self.additional_corrections_file_for_current_version
            self.dyn_lake_driver.current_date = slice_time
            os.mkdir(join(self.lakes_directory,
                          "results","diag_{}".format(slice_label)))
            self.dyn_lake_driver.no_intermediaries_dynamic_lake_driver()
            self.dyn_lake_driver.clean_work_dir()
            self.dyn_lake_driver.store_diagnostics(join(self.lakes_directory,
                                                   "results","diag_{}".format(slice_label)))
        if self.run_river_scripting or force_run_all:
            self.dyn_hd_driver.original_orography_filename=orography_filepath
            self.dyn_hd_driver.original_ls_mask_filename=landsea_mask_filepath
            self.dyn_hd_driver.output_hdparas_filepath=\
                join(self.rivers_directory,
                     "results","hdpara_{}.nc".format(slice_label))
            self.dyn_hd_driver.output_hdstart_filepath=\
                join(self.rivers_directory,
                             "results","hdstart_{}.nc".format(slice_label))
            self.dyn_hd_driver.glacier_mask_filename=glacier_mask_filepath
            self.dyn_hd_driver.non_standard_orog_correction_filename=\
                (self.base_corrections_filepath if self.corrections_version == 0 else
                 self.corrections_file_for_current_version)
            os.mkdir(join(self.rivers_directory,
                          "results","diag_{}".format(slice_label)))
            self.dyn_hd_driver.\
            no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
            self.dyn_hd_driver.clean_work_dir()
            self.dyn_hd_driver.store_diagnostics(join(self.rivers_directory,
                                                 "results","diag_{}".format(slice_label)))
        if self.run_hd_scripting_default_orography_corrections:
            if not os.path.isdir(join(self.rivers_directory,
                                      "results","default_orog_corrs")):
                os.mkdir(join(self.rivers_directory,
                              "results","default_orog_corrs"))
            old_python_config_filename = self.dyn_hd_driver.python_config_filename
            self.dyn_hd_driver.python_config_filename = join(self.ancillary_data_directory,
                                                             "default_orog_corrs_ancillaries",
                                                             "dynamic_hd_production_driver.cfg")
            self.dyn_hd_driver.original_orography_filename=orography_filepath
            self.dyn_hd_driver.original_ls_mask_filename=landsea_mask_filepath
            self.dyn_hd_driver.output_hdparas_filepath=\
                join(self.rivers_directory,
                     "results","default_orog_corrs","hdpara_{}.nc".format(slice_label))
            self.dyn_hd_driver.output_hdstart_filepath=\
                join(self.rivers_directory,
                     "results","default_orog_corrs","hdstart_{}.nc".format(slice_label))
            self.dyn_hd_driver.glacier_mask_filename=glacier_mask_filepath
            self.dyn_hd_driver.non_standard_orog_correction_filename=None
            os.mkdir(join(self.rivers_directory,
                              "results","default_orog_corrs","diag_{}".format(slice_label)))
            self.dyn_hd_driver.\
            no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
            self.dyn_hd_driver.clean_work_dir()
            self.dyn_hd_driver.store_diagnostics(join(self.rivers_directory,
                                                 "results","default_orog_corrs","diag_{}".format(slice_label)))
            self.dyn_hd_driver.python_config_filename = old_python_config_filename
            default_corrections_filepath = join(self.ancillary_data_directory,
                                                "default_orog_corrs_ancillaries",
                                                "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                "_glcc_olson_lsmask_0k_20170517_003802_with_grid.nc")
            if self.generate_present_day_rivers_with_original_sink_set and slice_time == 0 :
                os.mkdir(join(self.rivers_directory,"results","default_orog_corrs",
                              "diag_{}_original_truesinks".format(slice_label)))
                Dynamic_HD_Drivers.\
                    generate_present_day_river_directions_with_original_true_sink_set(orography_filepath,
                        landsea_mask_filepath,glacier_mask_filepath,
                        default_corrections_filepath,
                        output_dir=join(self.rivers_directory,"results","default_orog_corrs",
                                        "diag_{}_original_truesinks".format(slice_label)),
                        orography_corrections_fieldname="field_value")
            if self.generate_present_day_rivers_with_true_sinks and slice_time == 0 :
                true_sinks_filename = join(self.corrections_directory,"true_sinks_fields",
                                           "true_sinks_field_version_{}.nc".\
                                           format(self.corrections_version))
                os.mkdir(join(self.rivers_directory,"results","default_orog_corrs",
                              "diag_{}_with_truesinks".format(slice_label)))
                Dynamic_HD_Drivers.\
                    generate_present_day_river_directions_with_true_sinks(orography_filepath,
                        landsea_mask_filepath,glacier_mask_filepath,
                        default_corrections_filepath,
                        output_dir=join(self.rivers_directory,"results","default_orog_corrs",
                                        "diag_{}_with_truesinks".format(slice_label)),
                        truesinks_filepath=true_sinks_filename,
                        orography_corrections_fieldname="field_value")

        if self.generate_present_day_rivers_with_original_sink_set and slice_time == 0 :
            os.mkdir(join(self.rivers_directory,"results",
                          "diag_{}_original_truesinks".format(slice_label)))
            Dynamic_HD_Drivers.\
                generate_present_day_river_directions_with_original_true_sink_set(orography_filepath,
                    landsea_mask_filepath,glacier_mask_filepath,
                    (self.base_corrections_filepath if self.corrections_version == 0 else
                     self.corrections_file_for_current_version),
                     output_dir=join(self.rivers_directory,"results",
                                     "diag_{}_original_truesinks".format(slice_label)))
        if self.generate_present_day_rivers_with_true_sinks and slice_time == 0 :
            true_sinks_filename = join(self.corrections_directory,"true_sinks_fields",
                                       "true_sinks_field_version_{}.nc".\
                                       format(self.corrections_version))
            os.mkdir(join(self.rivers_directory,"results",
                          "diag_{}_with_truesinks".format(slice_label)))
            Dynamic_HD_Drivers.\
                generate_present_day_river_directions_with_true_sinks(orography_filepath,
                    landsea_mask_filepath,glacier_mask_filepath,
                    (self.base_corrections_filepath if self.corrections_version == 0 else
                     self.corrections_file_for_current_version),
                     output_dir=join(self.rivers_directory,"results",
                                     "diag_{}_with_truesinks".format(slice_label)),
                     truesinks_filepath=true_sinks_filename)

    def driver_transient_run(self,start_date=0,end_date=0,slice_spacing = 10,
                             dates_to_run_in=None):
        if not self.skip_current_day_time_slice:
            self.run_slice(0,force_run_all=False)#True)
        if start_date > end_date:
            dates_to_run = list(range(start_date,end_date,-slice_spacing))
        else:
            dates_to_run = []
        if dates_to_run_in != None:
            dates_to_run.extend(list(set(dates_to_run_in).\
                                     difference(set(dates_to_run))))
        if dates_to_run:
            for date in dates_to_run:
                self.run_slice(date)

    def run_setup_directory_structure(self):
        os.mkdir(self.base_directory)
        os.mkdir(self.lakes_directory)
        os.mkdir(join(self.lakes_directory,"work"))
        os.mkdir(join(self.lakes_directory,"results"))
        os.mkdir(self.rivers_directory)
        os.mkdir(join(self.rivers_directory,"work"))
        os.mkdir(join(self.rivers_directory,"results"))
        os.mkdir(self.corrections_directory)
        os.mkdir(join(self.corrections_directory,"work"))
        os.mkdir(join(self.corrections_directory,"correction_sets"))
        os.mkdir(join(self.corrections_directory,"date_based_correction_sets"))
        os.mkdir(join(self.corrections_directory,"additional_correction_sets"))
        os.mkdir(join(self.corrections_directory,"correction_fields"))
        os.mkdir(join(self.corrections_directory,"true_sinks_sets"))
        os.mkdir(join(self.corrections_directory,"true_sinks_fields"))
        with open(join(self.base_directory,"analysis_info.txt"),"w") as info_txt:
            info_txt.write("Creation date: {}\n".format(datetime.now()))
            info_txt.write("base corrections file: {}\n".format(self.base_corrections_filepath))
            info_txt.write("base date-based corrections file: {}\n".format(self.base_date_based_corrections_filepath))
            info_txt.write("base additional corrections file: {}\n".format(self.base_additional_corrections_filepath))
            info_txt.write("base true sinks file: {}\n".format(self.base_true_sinks_filepath))
            info_txt.write("ancillary data directory: {}\n".format(self.ancillary_data_directory))
            info_txt.write("present day base orography file: {}\n".\
                           format(self.present_day_base_orography_filepath))
            info_txt.write("orography file template: {}\n".\
                           format(self.orography_filepath_template))
            info_txt.write("landsea mask file template: {}\n".\
                           format(self.landsea_mask_filepath_template))
            info_txt.write("glacier mask file template: {}\n".\
                           format(self.glacier_mask_filepath_template))
        with open(join(self.base_directory,"versions.txt"),"w") as versions_txt:
            versions_txt.write("Version 0 - {}\n".format(datetime.now()))
        with open(join(self.corrections_directory,
                       "working_correction_set.txt"),"w") as corrections_txt:
            corrections_txt.write("# lat, lon, height")
        with open(join(self.corrections_directory,
                       "working_true_sinks_set.txt"),"w") as true_sinks_txt:
            true_sinks_txt.write("# lat, lon")
        if self.base_date_based_corrections_filepath is not None:
            shutil.copyfile(self.base_date_based_corrections_filepath,
                            join(self.corrections_directory,
                                 "working_date_based_correction_set.txt"))
            shutil.copyfile(self.base_date_based_corrections_filepath,
                            join(self.corrections_directory,"date_based_correction_sets",
                                 "date_based_corrections_set_version_0.txt"))
        if self.additional_based_corrections_filepath is not None:
            shutil.copyfile(self.additional_based_corrections_filepath,
                            join(self.corrections_directory,
                                 "working_additional_correction_set.txt"))
            shutil.copyfile(self.additional_based_corrections_filepath,
                            join(self.corrections_directory,"additional_correction_sets",
                                 "additional_corrections_set_version_0.txt"))

    def clear_lake_results(self,version):
        print("Clearing lake results...")
        if version == -1:
            for file in [ file for file in
                          os.listdir(join(self.lakes_directory,
                                                  "results")) if file.endswith(".nc")]:
                print("removing {}".format(join(self.lakes_directory,"results",file)))
                os.remove(join(self.lakes_directory,"results",file))
            for folder in [ folder for folder in
                            os.listdir(join(self.lakes_directory,
                                          "results")) if folder.startswith("diag_version_")]:
                for file in [ file for file in
                              os.listdir(join(self.lakes_directory,
                                          "results",folder)) if (file.endswith(".nc") or
                                                                 file.endswith(".txt"))]:
                    print("removing {}".format(join(self.lakes_directory,"results",folder,file)))
                    os.remove(join(self.lakes_directory,"results",folder,file))
                print("removing dir {}".format(join(self.lakes_directory,"results",folder)))
                os.rmdir(join(self.lakes_directory,"results",folder))
        else:
            for file in [ file for file in
                          os.listdir(join(self.lakes_directory,
                                                  "results")) if (file.endswith(".nc") and
                                                                  f'version_{version}' in file)]:
                print("removing {}".format(join(self.lakes_directory,"results",file)))
                os.remove(join(self.lakes_directory,"results",file))
            for folder in [ folder for folder in
                            os.listdir(join(self.lakes_directory,
                                          "results")) if folder.startswith(f'diag_version_{version}')]:
                for file in [ file for file in
                              os.listdir(join(self.lakes_directory,
                                          "results",folder)) if (file.endswith(".nc") or
                                                                 file.endswith(".txt"))]:
                    print("removing {}".format(join(self.lakes_directory,"results",folder,file)))
                    os.remove(join(self.lakes_directory,"results",folder,file))
                print("removing {}".format(join(self.lakes_directory,"results",folder)))
                os.rmdir(join(self.lakes_directory,"results",folder))

    def clear_river_results(self,version):
        print("Clearing river results...")
        if version == -1:
            for file in [ file for file in
                          os.listdir(join(self.rivers_directory,
                                          "results")) if file.endswith(".nc")]:
                print("removing {}".format(join(self.rivers_directory,"results",file)))
                os.remove(join(self.rivers_directory,"results",file))
            for folder in [ folder for folder in
                            os.listdir(join(self.rivers_directory,
                                          "results")) if folder.startswith("diag_version_")]:
                for file in [ file for file in
                              os.listdir(join(self.rivers_directory,
                                          "results",folder)) if (file.endswith(".nc") or
                                                                 file.endswith(".txt"))]:
                    print("removing {}".format(join(self.rivers_directory,"results",folder,file)))
                    os.remove(join(self.rivers_directory,"results",folder,file))
                print("removing {}".format(join(self.rivers_directory,"results",folder)))
                os.rmdir(join(self.rivers_directory,"results",folder))
        else:
            for file in [ file for file in
                          os.listdir(join(self.rivers_directory,
                                          "results")) if (file.endswith(".nc") and
                                                          f'version_{version}' in file)]:
                print("removing {}".format(join(self.rivers_directory,"results",file)))
                os.remove(join(self.rivers_directory,"results",file))
            for folder in [ folder for folder in
                            os.listdir(join(self.rivers_directory,
                                          "results")) if folder.startswith(f"diag_version_{version}")]:
                for file in [ file for file in
                              os.listdir(join(self.rivers_directory,
                                          "results",folder)) if (file.endswith(".nc") or
                                                                 file.endswith(".txt"))]:
                    print("removing {}".format(join(self.rivers_directory,"results",folder,file)))
                    os.remove(join(self.rivers_directory,"results",folder,file))
                print("removing {}".format(join(self.rivers_directory,"results",folder)))
                os.rmdir(join(self.rivers_directory,"results",folder))

    def clear_river_default_orog_corrs_results(self):
        print("Clearing default orography correction results...")
        for file in [ file for file in
                      os.listdir(join(self.rivers_directory,
                                      "results",
                                      "default_orog_corrs")) if file.endswith(".nc")]:
            os.remove(join(self.rivers_directory,"results","default_orog_corrs",file))
        for folder in [ folder for folder in
                        os.listdir(join(self.rivers_directory,"results",
                                        "default_orog_corrs")) if folder.startswith("diag_version_")]:
            for file in [ file for file in
                          os.listdir(join(self.rivers_directory,"results",
                                          "default_orog_corrs",folder)) if (file.endswith(".nc") or
                                                                 file.endswith(".txt"))]:
                os.remove(join(self.rivers_directory,"results","default_orog_corrs",folder,file))
            os.rmdir(join(self.rivers_directory,"results","default_orog_corrs",folder))

def setup_and_run_lake_analysis_from_command_line_arguments(args):
    """Setup and run tools for the dynamic lake analysis from command line arguments"""
    driver_object = Dynamic_Lake_Analysis_Run_Framework(**vars(args))
    driver_object.run_selected_processes()

class Arguments:
    """An empty class used to pass namelist arguments into the main routine as keyword arguments."""

    pass

def parse_arguments():
    """Parse the command line arguments using the argparse module.

    Returns:
    An Arguments object containing the comannd line arguments.
    """

    args = Arguments()
    parser = argparse.ArgumentParser("Run lake model analysis processes")
    parser.add_argument('base_directory',
                        metavar='base-directory',
                        help='Full path to base directory',
                        type=str)
    parser.add_argument('-s','--setup-directory-structure',
                        help='Setup a directory structure',
                        action="store_true")
    parser.add_argument("-A",'--ancillary-data-directory',
                        type=str)
    parser.add_argument("-P",'--present-day-base-orography-filepath',
                        type=str)
    parser.add_argument("-T",'--orography-filepath-template',
                        type=str)
    parser.add_argument("-L",'--landsea-mask-filepath-template',
                        type=str)
    parser.add_argument("-G",'--glacier-mask-filepath-template',
                        type=str)
    parser.add_argument("-c",'--base-corrections-filepath',type=str)
    parser.add_argument("-b",'--base-date-based-corrections-filepath',type=str,default=None)
    parser.add_argument("-u",'--base-true-sinks-filepath',type=str)
    parser.add_argument('-g','--generate-lake-orography-corrections',
                        action="store_true")
    parser.add_argument('-t','--apply-orography-tweaks',action="store_true")
    parser.add_argument('-y','--change-date-based-corrections',action="store_true")
    parser.add_argument('-r','--make_analysis_run',action="store_true")
    parser.add_argument('-S','--skip-dynamic-river-production',action="store_true")
    parser.add_argument('-K','--skip-dynamic-lake-production',action="store_true")
    parser.add_argument('-I','--skip-current-day-time-slice',action="store_true")
    parser.add_argument('-D','--run_hd_scripting_default_orography_corrections',
                        action="store_true")
    parser.add_argument('-d','--start-date',type=int,default=0)
    parser.add_argument('-e','--end-date',type=int,default=0)
    parser.add_argument('-i','--slice-spacing',type=int,default=10)
    parser.add_argument('--clear-lake-results',nargs='?',type=int,const=-1,default=-2)
    parser.add_argument('--clear-river-results',nargs='?',type=int,const=-1,default=-2)
    parser.add_argument('--clear-river-default-orog-corrs-results',
                        action="store_true")
    parser.add_argument('--generate-present-day-rivers-with-original-sink-set',
                        action="store_true")
    parser.add_argument('--generate-present-day-rivers-with-true-sinks',
                        action="store_true")
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    if args.setup_directory_structure and (args.ancillary_data_directory is None or
                                           args.base_corrections_filepath is None or
                                           args.base_true_sinks_filepath is None or
                                           args.present_day_base_orography_filepath is None or
                                           args.orography_filepath_template is None or
                                           args.landsea_mask_filepath_template is None or
                                           args.glacier_mask_filepath_template is None):
        parser.error("Option -s --setup-directory-structure requires "
                     "options -A --ancillary-data-directory, "
                     "-c --base-corrections-filepath, "
                     "-u --base-true-sinks-filepath, "
                     "-P --present-day-base-orography-filepath, "
                     "-T --orography-filepath-template, "
                     "-L --landsea-mask-filepath-template and "
                     "-G --glacier-mask-filepath-template")
    if ((args.generate_present_day_rivers_with_original_sink_set or
         args.generate_present_day_rivers_with_true_sinks) and
        args.skip_current_day_time_slice):
        raise RuntimeError("Incompatible options")
    if args.generate_lake_orography_corrections and args.apply_orography_tweaks:
        warnings.warn("Applying orography correction automatically applies"
                      "orography tweaks too, thus setting will"
                      "-t apply-orography-tweaks have no effect as "
                      "-g generate-lake-orography-corrections is set")
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_lake_analysis_from_command_line_arguments(args)
