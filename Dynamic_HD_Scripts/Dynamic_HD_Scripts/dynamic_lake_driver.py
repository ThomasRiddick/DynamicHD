'''
Created on Dec 4, 2017

@author: thomasriddick
'''
import dynamic_hd_driver
import dynamic_lake_operators
import fill_sinks_driver
import iodriver
import field
import numpy as np
import utilities
from os.path import join

class Dynamic_Lake_Drivers(dynamic_hd_driver.Dynamic_HD_Drivers):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(Dynamic_Lake_Drivers,self).__init__()

    def prepare_orography_ICE5G_0k_uncorrected(self):
        file_label = self._generate_file_label()
        ICE5G_0k_file = join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        ICE5G_0k_orography_fieldname = "orog"
        ICE5G_minima_filename = self.generated_minima_filepath+file_label+".nc"
        ICE5G_minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        ICE5G_minima_fieldname = "minima"
        ICE5G_flowdirs_filename = self.generated_rdir_filepath+file_label+'.nc'
        ICE5G_flowdirs_fieldname = "FDIR"
        ICE5G_output_orog_filename= self.generated_orography_filepath + file_label + '.nc'
        ICE5G_lakemask_filename= join(self.lakemask_filepath,"empty_lakemask.nc")
        ICE5G_lakemask_fieldname="lakemask"
        ICE5G_landsea_mask_filename=join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        ICE5G_landsea_mask_fieldname="sftlf"
        lakemask = field.makeEmptyField(field_type='Generic',dtype=np.int32,grid_type="LatLong10min")
        orog = iodriver.advanced_field_loader(ICE5G_0k_file,fieldname=ICE5G_0k_orography_fieldname)
        lakemask.grid = orog.get_grid()
        iodriver.advanced_field_writer(ICE5G_lakemask_filename,lakemask,
                                       ICE5G_lakemask_fieldname,clobber=True)

        dynamic_lake_operators.advanced_local_minima_finding_driver(ICE5G_0k_file,
                                                                    ICE5G_0k_orography_fieldname,
                                                                    ICE5G_minima_filename,
                                                                    ICE5G_minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(ICE5G_minima_filename,
                                                                ICE5G_minima_fieldname,
                                                                ICE5G_minima_reduced_filename,
                                                                ICE5G_minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=ICE5G_0k_file,
                                                                      output_filename=ICE5G_flowdirs_filename,
                                                                      ls_mask_filename=
                                                                      ICE5G_landsea_mask_filename,
                                                                      fieldname=ICE5G_0k_orography_fieldname,
                                                                      output_fieldname=
                                                                      ICE5G_flowdirs_fieldname,
                                                                      ls_mask_fieldname=
                                                                      ICE5G_landsea_mask_fieldname)
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  ICE5G_0k_file,
                                                                  input_orography_fieldname=
                                                                  ICE5G_0k_orography_fieldname,
                                                                  input_rdirs_file=
                                                                  ICE5G_flowdirs_filename,
                                                                  input_rdirs_fieldname=
                                                                  ICE5G_flowdirs_fieldname,
                                                                  input_minima_file=
                                                                  ICE5G_minima_reduced_filename,
                                                                  input_minima_fieldname=
                                                                  ICE5G_minima_fieldname,
                                                                  input_lakemask_file=
                                                                  ICE5G_lakemask_filename,
                                                                  input_lakemask_fieldname=
                                                                  ICE5G_lakemask_fieldname,
                                                                  output_orography_file=
                                                                  ICE5G_output_orog_filename,
                                                                  output_orography_fieldname=
                                                                  ICE5G_0k_orography_fieldname)

    def prepare_orography_ICE5G_0k_corrected(self):
        file_label = self._generate_file_label()
        original_orography_filename = join(self.orography_path,
                                                "ice5g_v1_2_00_0k_10min.nc")
        orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802.nc")
        intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "second_intermediary_" + file_label + '.nc'
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        output_orog_filename = self.generated_orography_filepath + "lake_" + file_label + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename=join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        original_landsea_mask_fieldname="sftlf"
        original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                    file_label + '_orig' + '.nc')
        original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                    file_label + '_grid' + '.nc')
        minima_filename = self.generated_minima_filepath+file_label+".nc"
        minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        minima_fieldname = "minima"
        lakemask_filename= self.lakemask_filepath+"empty_lakemask.nc"
        lakemask_fieldname="lakemask"
        glacial_mask_file = join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        utilities.change_dtype(input_filename=original_ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               input_fieldname=original_landsea_mask_fieldname,
                               output_fieldname="lsmask",
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=
                                              intermediary_orography_filename,
                                              original_orography_fieldname=\
                                              "orog",
                                              grid_type="LatLong10min")
        utilities.replace_corrected_orography_with_original_for_glaciated_grid_points_drivers(
          input_corrected_orography_file=intermediary_orography_filename,
          input_original_orography_file=original_orography_filename,
          input_glacier_mask_file=glacial_mask_file,
          out_orography_file=second_intermediary_orography_filename,
          grid_type="LatLong10min")
        iodriver.add_grid_information_to_field(target_filename=
                                               orography_filename,
                                               original_filename=
                                               second_intermediary_orography_filename,
                                               target_fieldname="field_value",
                                               original_fieldname="field_value",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        iodriver.add_grid_information_to_field(target_filename=
                                               original_ls_mask_with_grid_filename,
                                               original_filename=
                                               original_ls_mask_with_new_dtype_filename,
                                               target_fieldname="lsmask",
                                               original_fieldname="lsmask",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        dynamic_lake_operators.advanced_local_minima_finding_driver(orography_filename,
                                                                    "field_value",
                                                                    minima_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename,
                                                                minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=orography_filename,
                                                                      output_filename=rdirs_filename,
                                                                      ls_mask_filename=
                                                                      original_ls_mask_with_grid_filename,
                                                                      fieldname="field_value",
                                                                      output_fieldname=
                                                                      "rdir",
                                                                      ls_mask_fieldname=
                                                                      "lsmask")
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  orography_filename,
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
                                                                  output_orog_filename,
                                                                  output_orography_fieldname=
                                                                  "field_value")


    def prepare_orography_ICE6G_21k_corrected(self):
        file_label = self._generate_file_label()
        original_orography_filename = join(self.orography_path,
                                                "ice5g_v1_2_00_0k_10min.nc")
        ice6g_0k_filename = join(self.orography_path,
                                 "Ice6g_c_VM5a_10min_0k.nc")
        ice6g_21k_filename = join(self.orography_path,
                                 "Ice6g_c_VM5a_10min_21k.nc")
        orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802.nc")
        intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "second_intermediary_" + file_label + '.nc'
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        output_0k_ice5g_orog_filename = self.generated_orography_filepath + "0k_ice5g_lake_" + file_label + '.nc'
        output_21k_ice6g_orog_filename = self.generated_orography_filepath + "21k_ice6g_lake_" + file_label + '.nc'
        output_21k_ice6g_orog_sinkless_filename = (self.generated_orography_filepath +
                                                   "21k_ice6g_lake_sinkless_" + file_label + '.nc')
        output_21k_ice6g_orog_sinkless_improved_filename = (self.generated_orography_filepath +
                                                            "21k_ice6g_lake_sinkless_improved_" + file_label + '.nc')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename=join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        original_landsea_mask_fieldname="sftlf"
        original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                    file_label + '_orig' + '.nc')
        original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                    file_label + '_grid' + '.nc')
        minima_filename = self.generated_minima_filepath+file_label+".nc"
        minima_filename_21k = self.generated_minima_filepath+file_label+"_21k.nc"
        minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        minima_reduced_filename_21k = self.generated_minima_filepath+file_label+"_reduced_21k.nc"
        minima_fieldname = "minima"
        lakemask_filename= self.lakemask_filepath+"/empty_lakemask.nc"
        lakemask_fieldname="lakemask"
        glacial_mask_file = join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        utilities.change_dtype(input_filename=original_ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               input_fieldname=original_landsea_mask_fieldname,
                               output_fieldname="lsmask",
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=
                                              intermediary_orography_filename,
                                              original_orography_fieldname=\
                                              "orog",
                                              grid_type="LatLong10min")
        utilities.replace_corrected_orography_with_original_for_glaciated_grid_points_drivers(
          input_corrected_orography_file=intermediary_orography_filename,
          input_original_orography_file=original_orography_filename,
          input_glacier_mask_file=glacial_mask_file,
          out_orography_file=second_intermediary_orography_filename,
          grid_type="LatLong10min")
        iodriver.add_grid_information_to_field(target_filename=
                                               orography_filename,
                                               original_filename=
                                               second_intermediary_orography_filename,
                                               target_fieldname="field_value",
                                               original_fieldname="field_value",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        iodriver.add_grid_information_to_field(target_filename=
                                               original_ls_mask_with_grid_filename,
                                               original_filename=
                                               original_ls_mask_with_new_dtype_filename,
                                               target_fieldname="lsmask",
                                               original_fieldname="lsmask",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        dynamic_lake_operators.advanced_local_minima_finding_driver(orography_filename,
                                                                    "field_value",
                                                                    minima_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename,
                                                                minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=orography_filename,
                                                                      output_filename=rdirs_filename,
                                                                      ls_mask_filename=
                                                                      original_ls_mask_with_grid_filename,
                                                                      fieldname="field_value",
                                                                      output_fieldname=
                                                                      "rdir",
                                                                      ls_mask_fieldname=
                                                                      "lsmask")
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  orography_filename,
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
                                                                  output_0k_ice5g_orog_filename,
                                                                  output_orography_fieldname=
                                                                  "Topo")
        utilities.advanced_rebase_orography_driver(orography_filename=
                                                   ice6g_21k_filename,
                                                   present_day_base_orography_filename=
                                                   ice6g_0k_filename,
                                                   present_day_reference_orography_filename=
                                                   output_0k_ice5g_orog_filename,
                                                   rebased_orography_filename=
                                                   output_21k_ice6g_orog_filename,
                                                   orography_fieldname="Topo",
                                                   present_day_base_orography_fieldname="Topo",
                                                   present_day_reference_orography_fieldname="Topo",
                                                   rebased_orography_fieldname="Topo")
        fill_sinks_driver.\
        generate_orography_with_sinks_filled_advanced_driver(output_21k_ice6g_orog_filename,
                                                             output_21k_ice6g_orog_sinkless_filename,
                                                             "Topo",
                                                             "Topo",
                                                             ls_mask_filename=None,
                                                             truesinks_filename=None,
                                                             ls_mask_fieldname=None,
                                                             truesinks_fieldname=None,
                                                             add_slight_slope_when_filling_sinks=False,
                                                             slope_param=0.1)
        ice6g_sinkless_field = iodriver.advanced_field_loader(output_21k_ice6g_orog_sinkless_filename,
                                                              fieldname="Topo",
                                                              adjust_orientation=True)
        ice6g_field = iodriver.advanced_field_loader(output_21k_ice6g_orog_filename,fieldname="Topo",
                                                     adjust_orientation=True)
        ice6g_21k_icemask = iodriver.advanced_field_loader(ice6g_21k_filename,
                                                           fieldname="sftgif",
                                                           adjust_orientation=True)
        ice6g_21k_lsmask = iodriver.advanced_field_loader(ice6g_21k_filename,
                                                          fieldname="sftlf",
                                                          adjust_orientation=True)
        ice6g_21k_icemask.invert_data()
        ice6g_field.mask_field_with_external_mask(ice6g_21k_icemask.get_data())
        ice6g_sinkless_field.update_field_with_partially_masked_data(ice6g_field)
        ice6g_field.mask_field_with_external_mask(ice6g_21k_lsmask.get_data())
        ice6g_sinkless_field.update_field_with_partially_masked_data(ice6g_field)
        iodriver.advanced_field_writer(output_21k_ice6g_orog_sinkless_improved_filename,
                                       ice6g_sinkless_field,fieldname="Topo",clobber=True)
        dynamic_lake_operators.advanced_local_minima_finding_driver(output_21k_ice6g_orog_filename,
                                                                    "Topo",
                                                                    minima_filename_21k,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename_21k,
                                                                minima_fieldname,
                                                                minima_reduced_filename_21k,
                                                                minima_fieldname)
        dynamic_lake_operators.\
          advanced_basin_evaluation_driver(input_minima_file=minima_reduced_filename_21k,
                                           input_minima_fieldname=minima_fieldname,
                                           input_raw_orography_file=ice6g_21k_filename,
                                           input_raw_orography_fieldname="Topo",
                                           input_corrected_orography_file=output_21k_ice6g_orog_filename,
                                           input_corrected_orography_fieldname="Topo",
                                           input_prior_fine_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/""updated_RFDs_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_""rdirs_tarasov_orog_corrs_20171015_031541.nc",
                                           input_prior_fine_rdirs_fieldname="field_value",
                                           input_prior_fine_catchments_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_unsorted_ICE6g_lgm_ALG4_sinkless_no_true_sinks"
                                           "_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20171015_031541.nc",
                                           input_prior_fine_catchments_fieldname="field_value",
                                           input_coarse_catchment_nums_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks"
                                           "_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs"
                                           "_20171015_031541_upscaled_updated.nc",
                                           input_coarse_catchment_nums_fieldname="field_value")

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """
    lake_drivers = Dynamic_Lake_Drivers()
    #lake_drivers.prepare_orography_ICE5G_0k_uncorrected()
    #lake_drivers.prepare_orography_ICE5G_0k_corrected()
    lake_drivers.prepare_orography_ICE6G_21k_corrected()

if __name__ == '__main__':
    main()
