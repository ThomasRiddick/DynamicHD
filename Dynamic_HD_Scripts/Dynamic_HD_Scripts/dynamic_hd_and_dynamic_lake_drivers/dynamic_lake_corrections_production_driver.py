import numpy as np
import os
from os.path import join
import os.path as path
import time
import datetime
import re
import glob
from enum import Enum
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.tools import determine_river_directions
from Dynamic_HD_Scripts.tools import fill_sinks_driver
from Dynamic_HD_Scripts.tools import compute_catchments as cc
from Dynamic_HD_Scripts.tools import flow_to_grid_cell as ftgc
from Dynamic_HD_Scripts.tools import dynamic_lake_operators
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers import dynamic_hd_driver as dyn_hd_dr

class CorrectionTypes(Enum):
    PRELIMINARY = 1
    FINAL = 2

class Dynamic_Lake_Correction_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):

    first_line_pattern = re.compile(r"^\s*#\s*lat\s*,\s*lon\s*,\s*height$")
    true_sinks_first_line_pattern = re.compile(r"^\s*#\s*lat\s*,\s*lon\s*$")
    prelim_match = re.compile(r"^\s*#\s*Prelim",flags=re.IGNORECASE)
    final_match = re.compile(r"^\s*#\s*Final",flags=re.IGNORECASE)
    comment_line_match = re.compile(r"\s*#")
    remove_sink_line_match = re.compile(r"\s*R\s*,\s*[0-9]*\s*,\s*[0-9]*\s*")

    def __init__(self,working_directory=None):
        super(Dynamic_Lake_Correction_Production_Run_Drivers,self).__init__()
        self.working_directory = working_directory

    def no_intermediaries_lake_corrections_driver(self,
                                                  version=None,
                                                  original_orog_corrections_filename=None,
                                                  new_orography_corrections_filename=None,
                                                  true_sinks_filename=None):
        tstart = time.time()
        if version is None:
            file_label = self._generate_file_label()
        else:
            file_label = "{}_{}".format(version,datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        ls_mask_filename = join(self.ls_masks_path,"generated",
                                "ls_mask_make_1000m_depth_contour_mask_from_ICE6G_20200721_144332.nc")
        ls_mask_fieldname = "lsm"
        original_orography_filename = join(self.orography_path,
                                           "ice5g_v1_2_00_0k_10min.nc")
        if true_sinks_filename is None:
            true_sinks_filename = join(self.truesinks_path,
                                       "truesinks_ICE5G_and_tarasov_upscaled_srtm30plus_"
                                       "north_america_only_data_ALG4_sinkless_glcc_olson"
                                       "_lsmask_0k_20191014_173825_with_grid.nc")
        if original_orog_corrections_filename is None:
            original_orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                      "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                      "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                      "_glcc_olson_lsmask_0k_20170517_003802_g.nc")
        input_bathymetry_file=join(self.lake_bathymetry_filepath,"NOAA_great_lakes_bathymetry.nc")
        input_bathymetry_fieldname="Band1"
        lake_mask_file=join(self.lakemask_filepath,"NOAA_great_lakes_mask.nc")
        lake_mask_fieldname="lakemask"
        dummy_lake_mask_filename= self.lakemask_filepath+"/empty_lakemask.nc"
        dummy_lake_mask_fieldname="lakemask"
        minima_fieldname = "minima"
        glacier_mask_filename = join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        if self.working_directory is None:
            intermediary_orography_filename = self.generated_orography_filepath +\
                                                    "intermediary_" + file_label + '.nc'
            second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                    "second_intermediary_" + file_label + '.nc'
            third_intermediary_orography_filename = self.generated_orography_filepath +\
                                                    "third_intermediary_" + file_label + '.nc'
            output_0k_ice5g_orog_filename = self.generated_orography_filepath + \
                                            "0k_ice5g_lake_" + file_label + '.nc'
            rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
            original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                        file_label + '_orig' + '.nc')
            original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                        file_label + '_grid' + '.nc')
            minima_filename = self.generated_minima_filepath+file_label+".nc"
            minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        else:
            intermediary_orography_filename = join(self.working_directory,
                                                   "intermediary_orog_" + file_label + '.nc')
            second_intermediary_orography_filename = join(self.working_directory,
                                                          "second_intermediary_orog_" +
                                                          file_label + '.nc')
            third_intermediary_orography_filename = join(self.working_directory,
                                                         "third_intermediary_orog_" +
                                                         file_label + '.nc')
            output_0k_ice5g_orog_filename = join(self.working_directory,
                                                 "orog_0k_with_lakecorrs_" +
                                                 file_label + '.nc')
            rdirs_filename = join(self.working_directory, "rdirs_" + file_label + '.nc')
            original_ls_mask_with_new_dtype_filename = join(self.working_directory,
                                                            "lsmask_" + file_label +
                                                            '_orig' + '.nc')
            original_ls_mask_with_grid_filename= join(self.working_directory,
                                                      "lsmask_" + file_label + '_grid' + '.nc')
            minima_filename = join(self.working_directory, "minima_" +file_label+".nc")
            minima_reduced_filename = join(self.working_directory, "minima_" +
                                           file_label + "_reduced.nc")
        if new_orography_corrections_filename is None:
            new_orography_corrections_filename = path.join(self.orography_corrections_fields_path,
                                                           "ice5g_0k_lake_corrs_" + file_label + ".nc")
        utilities.change_dtype(input_filename=ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               input_fieldname=ls_mask_fieldname,
                               output_fieldname="lsmask",
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.advanced_apply_orog_correction_field(original_orography_filename=
                                                       original_orography_filename,
                                                       orography_corrections_filename=
                                                       original_orog_corrections_filename,
                                                       corrected_orography_filename=
                                                       intermediary_orography_filename,
                                                       original_orography_fieldname=
                                                       "orog")
        # utilities.advanced_replace_corrected_orog_with_orig_for_glcted_grid_points_drivers(
        #   input_corrected_orography_file=intermediary_orography_filename,
        #   input_original_orography_file=original_orography_filename,
        #   input_glacier_mask_file=glacier_mask_filename,
        #   out_orography_file=second_intermediary_orography_filename,
        #   input_corrected_orography_fieldname=None,
        #   input_original_orography_fieldname=None,
        #   input_glacier_mask_fieldname="sftgif",
        #   out_orography_fieldname=None)
        second_intermediary_orography_filename = intermediary_orography_filename
        iodriver.add_grid_information_to_field(target_filename=
                                               original_ls_mask_with_grid_filename,
                                               original_filename=
                                               original_ls_mask_with_new_dtype_filename,
                                               target_fieldname="lsmask",
                                               original_fieldname="lsmask",
                                               flip_ud_raw=False,rotate180lr_raw=False,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        dynamic_lake_operators.advanced_local_minima_finding_driver(second_intermediary_orography_filename,
                                                                    "field_value",
                                                                    minima_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename,
                                                                minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=second_intermediary_orography_filename,
                                                                      output_filename=rdirs_filename,
                                                                      ls_mask_filename=
                                                                      original_ls_mask_with_grid_filename,
                                                                      truesinks_filename=
                                                                      true_sinks_filename,
                                                                      fieldname="field_value",
                                                                      output_fieldname=
                                                                      "rdir",
                                                                      ls_mask_fieldname=
                                                                      "lsmask",
                                                                      truesinks_fieldname=
                                                                      "true_sinks")
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  second_intermediary_orography_filename,
                                                                  input_orography_fieldname=
                                                                  "field_value",
                                                                  input_rdirs_file=
                                                                  rdirs_filename,
                                                                  input_rdirs_fieldname=
                                                                  "rdir",
                                                                  input_minima_file=
                                                                  minima_filename,
                                                                  input_minima_fieldname=
                                                                  minima_fieldname,
                                                                  input_lakemask_file=
                                                                  dummy_lake_mask_filename,
                                                                  input_lakemask_fieldname=
                                                                  dummy_lake_mask_fieldname,
                                                                  output_orography_file=
                                                                  third_intermediary_orography_filename,
                                                                  output_orography_fieldname=
                                                                  "Topo",
                                                                  add_slope = True,
                                                                  max_exploration_range = 10,
                                                                  minimum_height_change_threshold = 5.0,
                                                                  short_path_threshold = 6,
                                                                  short_minimum_height_change_threshold = 0.25)
        dynamic_lake_operators.add_lake_bathymetry_driver(input_orography_file=
                                                          third_intermediary_orography_filename,
                                                          input_orography_fieldname="Topo",
                                                          input_bathymetry_file=
                                                          input_bathymetry_file,
                                                          input_bathymetry_fieldname=
                                                          input_bathymetry_fieldname,
                                                          lake_mask_file=lake_mask_file,
                                                          lake_mask_fieldname=
                                                          lake_mask_fieldname,
                                                          output_orography_file=
                                                          output_0k_ice5g_orog_filename,
                                                          output_orography_fieldname="Topo")
        utilities.advanced_orog_correction_field_generator(original_orography_filename=
                                                           original_orography_filename,
                                                           corrected_orography_filename=
                                                           output_0k_ice5g_orog_filename,
                                                           orography_corrections_filename=
                                                           new_orography_corrections_filename,
                                                           original_orography_fieldname=
                                                           "orog",
                                                           corrected_orography_fieldname=
                                                           "Topo",
                                                           orography_corrections_fieldname=
                                                           "orog")
        print("Time for correction generation: " + str(time.time() - tstart))

    def apply_tweaks(self,
                     initial_corrections_filename,
                     output_corrections_filename,
                     corrections_list_filename,
                     corrections_type):
        if self.working_directory is None:
            raise RuntimeError("Applying tweaks requires a working directory to be specified")
        print("Note - All adjustments must be relative to a 180 degree W to 180 degree E,"
              "90 degree N to 90 degree S grid")
        original_orography_filename = join(self.orography_path,
                                           "ice5g_v1_2_00_0k_10min.nc")
        intermediary_orography_filename = join(self.working_directory,
                                               "pre_{}_tweak_orography.nc".\
                                               format("final" if
                                                      corrections_type == CorrectionTypes.FINAL else
                                                      "preliminary"))
        corrected_orography_filename = join(self.working_directory,
                                            "post_{}_tweak_orography.nc".\
                                            format("final" if
                                                   corrections_type == CorrectionTypes.FINAL else
                                                   "preliminary"))
        utilities.advanced_apply_orog_correction_field(original_orography_filename=
                                                       original_orography_filename,
                                                       orography_corrections_filename=
                                                       initial_corrections_filename,
                                                       corrected_orography_filename=
                                                       intermediary_orography_filename,
                                                       original_orography_fieldname="orog",
                                                       orography_corrections_fieldname="orog",
                                                       corrected_orography_fieldname="orog")
        orography_field =  iodriver.advanced_field_loader(intermediary_orography_filename,
                                                          field_type='Orography',
                                                          fieldname="orog")
        read_lines_from_file=False
        corrections_list = []
        with open(corrections_list_filename,"r") as corr_list_file:
            if not self.first_line_pattern.match(corr_list_file.readline().strip()):
                raise RuntimeError("List of corrections being loaded has"
                                   " incorrect header format")
            for line in corr_list_file:
                if self.prelim_match.match(line):
                    if corrections_type == CorrectionTypes.PRELIMINARY:
                        read_lines_from_file=True
                    else:
                        read_lines_from_file=False
                elif self.final_match.match(line):
                    if corrections_type == CorrectionTypes.FINAL:
                        read_lines_from_file=True
                    else:
                        read_lines_from_file=False
                elif self.comment_line_match.match(line):
                    continue
                elif read_lines_from_file:
                    corrections_list.append(tuple(int(value) if i < 2 else float(value) \
                                            for i,value in enumerate(line.strip().split(","))))
        for lat,lon,height in corrections_list:
            print("Correcting height of lat={0},lon={1} to {2} m".format(lat,lon,height))
            orography_field.get_data()[lat,lon] = height
        iodriver.advanced_field_writer(corrected_orography_filename,
                                       orography_field,
                                       fieldname="orog")
        utilities.advanced_orog_correction_field_generator(original_orography_filename=
                                                           original_orography_filename,
                                                           corrected_orography_filename=
                                                           corrected_orography_filename,
                                                           orography_corrections_filename=
                                                           output_corrections_filename,
                                                           original_orography_fieldname=
                                                           "orog",
                                                           corrected_orography_fieldname=
                                                           "orog",
                                                           orography_corrections_fieldname=
                                                           "orog")

    def prepare_true_sinks(self,initial_true_sinks_filename,
                           output_true_sinks_filename,
                           true_sinks_list_filename):
        if self.working_directory is None:
            raise RuntimeError("Applying tweaks requires a working directory to be specified")
        print("Note - All adjustments must be relative to a 180 degree W to 180 degree E,"
              "90 degree N to 90 degree S grid")
        true_sinks_field =  iodriver.advanced_field_loader(initial_true_sinks_filename,
                                                           field_type='Generic',
                                                           fieldname="true_sinks")
        true_sinks_list = []
        true_sinks_to_remove_list = []
        with open(true_sinks_list_filename,"r") as true_sinks_list_file:
            if not self.true_sinks_first_line_pattern.\
                match(true_sinks_list_file.readline().strip()):
                raise RuntimeError("List of extra true sinks being loaded has"
                                   " incorrect header format")
            for line in true_sinks_list_file:
                if self.comment_line_match.match(line):
                    continue
                elif self.remove_sink_line_match.match(line):
                    true_sinks_to_remove_list.append(tuple(int(value)
                                                           for value in line.strip().split(",")[1:]))
                else:
                    true_sinks_list.append(tuple(int(value) for value in line.strip().split(",")))
        for lat,lon in true_sinks_list:
            print("Adding true sinks at lat={0},lon={1}".format(lat,lon))
            true_sinks_field.get_data()[lat,lon] = True
        for lat,lon in true_sinks_to_remove_list:
            print("Removing true sinks at lat={0},lon={1}".format(lat,lon))
            true_sinks_field.get_data()[lat,lon] = False
        iodriver.advanced_field_writer(output_true_sinks_filename,
                                       true_sinks_field,
                                       fieldname="true_sinks")

    def clean_work_dir(self,partial_clean=False):
        templates = ["lsmask_*_orig.nc","lsmask_*_grid.nc",
                     "minima_*.nc","intermediary_orog_*.nc",
                     "third_intermediary_orog_*.nc",
                     "orog_0k_with_lakecorrs_*.nc",
                     "rdirs_*.nc"]
        files_to_remove = []
        for template in templates:
            files_to_remove.extend(glob.glob(join(self.working_directory,template)))
        for file in files_to_remove:
            os.remove(file)
        if not partial_clean:
            if path.exists(join(self.working_directory,"pre_final_tweak_orography.nc")):
                os.remove(join(self.working_directory,"pre_final_tweak_orography.nc"))
            if path.exists(join(self.working_directory,"pre_preliminary_tweak_orography.nc")):
                os.remove(join(self.working_directory,"pre_preliminary_tweak_orography.nc"))
            if path.exists(join(self.working_directory,"post_final_tweak_orography.nc")):
                os.remove(join(self.working_directory,"post_final_tweak_orography.nc"))
            if path.exists(join(self.working_directory,"post_preliminary_tweak_orography.nc")):
                os.remove(join(self.working_directory,"post_preliminary_tweak_orography.nc"))

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """
    lake_correction_drivers = Dynamic_Lake_Correction_Production_Run_Drivers()
    lake_correction_drivers.no_intermediaries_lake_corrections_driver()

if __name__ == '__main__':
    main()
