from os import path
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import iohelper
from Dynamic_HD_Scripts.base.field import Field
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers \
    import dynamic_hd_driver

class Paragen_Data_Extractor(object):

    def extract_data_to_run_paragen_model(self,
                                          rdirs_filename,
                                          orography_filename,
                                          variance_filename,
                                          innerslope_filename,
                                          glacier_filename,
                                          landsea_filename,
                                          cell_areas_filename,
                                          hdpara_filename,
                                          section_coords,
                                          language="Julia",
                                          write_to_text_file_filename=None):
        print("Extracting section:")
        print("min lat: {}".format(section_coords["min_lat"]))
        print("max lat: {}".format(section_coords["max_lat"]))
        print("min lon: {}".format(section_coords["min_lon"]))
        print("max lon: {}".format(section_coords["max_lon"]))
        extracted_data = ""
        rdirs = iodriver.advanced_field_loader(rdirs_filename,
                                               field_type='Generic',
                                               fieldname="field_value")
        extracted_data += rdirs.extract_data(section_coords,"rdirs",
                                             "integer",language)
        orography = iodriver.advanced_field_loader(orography_filename,
                                                   field_type='Generic',
                                                   fieldname="field_value")
        extracted_data += orography.extract_data(section_coords,"orography",
                                                 "double",language)
        glacier_mask = iodriver.advanced_field_loader(glacier_filename,
                                                      field_type='Generic',
                                                      fieldname="field_value")
        extracted_data += glacier_mask.extract_data(section_coords,"glacier_mask",
                                                    "integer",language)
        landsea_mask = iodriver.advanced_field_loader(landsea_filename,
                                                      field_type='Generic',
                                                      fieldname="field_value")
        extracted_data += landsea_mask.extract_data(section_coords,"lsmask",
                                                    "integer",language)
        innerslope = iodriver.advanced_field_loader(innerslope_filename,
                                                    field_type='Generic',
                                                    fieldname="field_value")
        extracted_data += innerslope.extract_data(section_coords,"innerslope",
                                                  "double",language)
        variance = iodriver.advanced_field_loader(variance_filename,
                                                  field_type='Generic',
                                                  fieldname="field_value")
        extracted_data += variance.extract_data(section_coords,"variance",
                                                "double",language)
        #Expected output data
        expected_rdirs = iodriver.advanced_field_loader(hdpara_filename,
                                                        field_type="Generic",
                                                        fieldname="FDIR")
        extracted_data += expected_rdirs.extract_data(section_coords,"expected_rdirs",
                                                      "double",language)
        expected_flag = iodriver.advanced_field_loader(hdpara_filename,
                                                       field_type="Generic",
                                                       fieldname="FLAG")
        extracted_data += expected_flag.extract_data(section_coords,"expected_flag",
                                                     "integer",language)
        expected_alf_k = iodriver.advanced_field_loader(hdpara_filename,
                                                        field_type="Generic",
                                                        fieldname="ALF_K")
        extracted_data += expected_alf_k.extract_data(section_coords,"expected_alf_k",
                                                      "double",language)
        expected_alf_n = iodriver.advanced_field_loader(hdpara_filename,
                                                        field_type="Generic",
                                                        fieldname="ALF_N")
        extracted_data += expected_alf_n.extract_data(section_coords,"expected_alf_n",
                                                      "integer",language)
        expected_arf_k = iodriver.advanced_field_loader(hdpara_filename,
                                                        field_type="Generic",
                                                        fieldname="ARF_K")
        extracted_data += expected_arf_k.extract_data(section_coords,"expected_arf_k",
                                                      "double",language)
        expected_arf_n = iodriver.advanced_field_loader(hdpara_filename,
                                                        field_type="Generic",
                                                        fieldname="ARF_N")
        extracted_data += expected_arf_n.extract_data(section_coords,"expected_arf_n",
                                                      "integer",language)
        expected_agf_k = iodriver.advanced_field_loader(hdpara_filename,
                                                        field_type="Generic",
                                                        fieldname="AGF_K")
        extracted_data += expected_agf_k.extract_data(section_coords,"expected_agf_k",
                                                      "double",language)
        cell_areas = iohelper.NetCDF4FileIOHelper.load_field(hdpara_filename,unmask=True,timeslice=None,
                                                             fieldname='AREA',check_for_grid_info=False,
                                                             grid_info=None,grid_type='Generic1D')
        extracted_data += Field(cell_areas,grid="Generic1D").\
                                extract_data({"min":section_coords["min_lat"],
                                              "max":section_coords["max_lat"]},
                                              "cell_areas",
                                              "double",language)
        if write_to_text_file_filename:
            print("Writing output to {}".format(write_to_text_file_filename))
            with open(write_to_text_file_filename, "w") as output_file:
                output_file.write(extracted_data)
        else:
            print(extracted_data)

def main():
    paragen_data_base_path = "/Users/thomasriddick/Documents/data/temp/paragen_test_data"
    extract_data_to_run_paragen_model = Paragen_Data_Extractor()
    extract_data_to_run_paragen_model.\
        extract_data_to_run_paragen_model(rdirs_filename=path.join(paragen_data_base_path,
                                                                   "30min_rdirs.nc"),
                                          orography_filename=path.join(paragen_data_base_path,
                                                                       "30minute_filled_orog_temp.nc"),
                                          variance_filename=path.join(paragen_data_base_path,
                                                                      "bin_toposig.nc"),
                                          innerslope_filename=path.join(paragen_data_base_path,
                                                                        "innerslope.nc"),
                                          glacier_filename=path.join(paragen_data_base_path,
                                                                     "null.nc"),
                                          landsea_filename=path.join(paragen_data_base_path,
                                                                     "30minute_ls_mask_temp.nc"),
                                          cell_areas_filename=path.join(paragen_data_base_path,
                                                                         ),
                                          hdpara_filename=path.join(paragen_data_base_path,
                                                                    "hdpara_trial_run_using_data_from_"
                                                                    "new_data_from_virna_2016_version_"
                                                                    "20221009_114734.nc"),
                                          section_coords={"min_lat":90,
                                                          "max_lat":110,
                                                          "min_lon":200,
                                                          "max_lon":220},
                                          language="Julia",
                                          write_to_text_file_filename=None)
                                          #write_to_text_file_filename="/Users/thomasriddick/Documents/"
                                          #                          "data/temp/extracted_data.txt")

if __name__ == '__main__':
    main()
