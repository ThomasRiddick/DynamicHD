import dynamic_hd_driver as dyn_hd_dr
import determine_river_directions
import dynamic_hd_driver
import dynamic_lake_operators
import fill_sinks_driver
import iodriver
import field
import numpy as np
import utilities
import compute_catchments as cc
import flow_to_grid_cell as ftgc
from os.path import join
import os.path as path
import time

class Dynamic_Lake_Correction_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):

    def no_intermediaries_lake_corrections_driver(self):
        tstart = time.time()
        file_label = self._generate_file_label()
        ls_mask_filename = join(self.ls_masks_path,"generated",
                                "ls_mask_make_1000m_depth_contour_mask_from_ICE6G_20200721_144332.nc")
        ls_mask_fieldname = "lsm"
        original_orography_filename = join(self.orography_path,
                                                "ice5g_v1_2_00_0k_10min.nc")
        true_sinks_filename = join(self.truesinks_path,
                                   "truesinks_ICE5G_and_tarasov_upscaled_srtm30plus_"
                                   "north_america_only_data_ALG4_sinkless_glcc_olson"
                                   "_lsmask_0k_20191014_173825_with_grid.nc")
        orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802_g.nc")
        input_bathymetry_file=join(self.lake_bathymetry_filepath,"NOAA_great_lakes_bathymetry.nc")
        input_bathymetry_fieldname="Band1"
        lake_mask_file=join(self.lakemask_filepath,"NOAA_great_lakes_mask.nc")
        lake_mask_fieldname="lakemask"
        intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "second_intermediary_" + file_label + '.nc'
        third_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "third_intermediary_" + file_label + '.nc'
        output_0k_ice5g_orog_filename = self.generated_orography_filepath + "0k_ice5g_lake_" + file_label + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                    file_label + '_orig' + '.nc')
        original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                    file_label + '_grid' + '.nc')
        minima_filename = self.generated_minima_filepath+file_label+".nc"
        minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        minima_fieldname = "minima"
        lakemask_filename= self.lakemask_filepath+"/empty_lakemask.nc"
        lakemask_fieldname="lakemask"
        glacier_mask_filename = join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
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
                                                       orog_corrections_filename,
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
                                                                  lakemask_filename,
                                                                  input_lakemask_fieldname=
                                                                  lakemask_fieldname,
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
        print "Time for initial setup: " + str(time.time() - tstart)

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """
    lake_correction_drivers = Dynamic_Lake_Correction_Production_Run_Drivers()
    lake_correction_drivers.no_intermediaries_lake_corrections_driver()

if __name__ == '__main__':
    main()
