from . import dynamic_hd_driver
from . import iodriver
import math

class Lake_Data_Extractor(object):

    def extract_data_to_run_lake_model(self,
                                       lake_parameters_filepath,
                                       hd_parameters_filepath,
                                       section_coords,
                                       language="Fortran",
                                       scale_factor=3.0,
                                       lake_initial_conditions_filepath=None,
                                       hd_initial_conditions_filepath=None,
                                       write_to_text_file_filename=None):
        extracted_data = ""
        if section_coords["min_lat"]%scale_factor != 0.0:
              section_coords["min_lat"] = int(math.ceil(section_coords["min_lat"]/3.0))*3
        if section_coords["min_lon"]%scale_factor != 0.0:
              section_coords["min_lon"] = int(math.ceil(section_coords["min_lon"]/3.0))*3
        if (section_coords["max_lat"]+1)%scale_factor != 0.0:
              section_coords["max_lat"] = int(math.floor(section_coords["max_lat"]/3.0))*3 - 1
        if (section_coords["max_lon"]+1)%scale_factor != 0.0:
              section_coords["max_lon"] = int(math.floor(section_coords["max_lon"]/3.0))*3 - 1
        coarse_section_coords = {"min_lat":int(math.floor(section_coords["min_lat"]/scale_factor)),
                                 "max_lat":int(math.floor(section_coords["max_lat"]/scale_factor)),
                                 "min_lon":int(math.floor(section_coords["min_lon"]/scale_factor)),
                                 "max_lon":int(math.floor(section_coords["max_lon"]/scale_factor))}
        print("Extracting section:")
        print("min lat: {}".format(section_coords["min_lat"]))
        print("max lat: {}".format(section_coords["max_lat"]))
        print("min lon: {}".format(section_coords["min_lon"]))
        print("max lon: {}".format(section_coords["max_lon"]))
        print("Extracting course section:")
        print("min lat: {}".format(coarse_section_coords["min_lat"]))
        print("max lat: {}".format(coarse_section_coords["max_lat"]))
        print("min lon: {}".format(coarse_section_coords["min_lon"]))
        print("max lon: {}".format(coarse_section_coords["max_lon"]))
        merge_points = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                      field_type='Generic',
                                                      fieldname="merge_points")
        extracted_data += merge_points.extract_data(section_coords,"merge_points",
                                                    "integer",language)
        lake_centers = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                      field_type='Generic',
                                                      fieldname="lake_centers")
        extracted_data += lake_centers.extract_data(section_coords,"lake_centers",
                                                  "integer",language)
        flood_volume_thresholds = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_volume_thresholds")
        extracted_data += flood_volume_thresholds.\
            extract_data(section_coords,"flood_volume_thresholds",
                         "double",language)
        flood_redirect_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_redirect_lat_index")
        extracted_data += flood_redirect_lat_index.\
            extract_data(section_coords,"flood_redirect_lat_index","integer",language)
        flood_redirect_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_redirect_lon_index")
        extracted_data += flood_redirect_lon_index.\
            extract_data(section_coords,"flood_redirect_lon_index","integer",language)
        flood_next_cell_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_next_cell_lat_index")
        extracted_data += flood_next_cell_lat_index.\
            extract_data(section_coords,"flood_next_cell_lat_index","integer",language)
        flood_next_cell_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_next_cell_lon_index")
        extracted_data += flood_next_cell_lon_index.\
            extract_data(section_coords,"flood_next_cell_lon_index","integer",language)
        flood_local_redirect = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_local_redirect")
        extracted_data += flood_local_redirect.\
            extract_data(section_coords,"flood_local_redirect","integer",language)
        flood_force_merge_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_force_merge_lat_index")
        extracted_data += flood_force_merge_lat_index.\
            extract_data(section_coords,"flood_force_merge_lat_index","integer",language)
        flood_force_merge_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="flood_force_merge_lon_index")
        extracted_data += flood_force_merge_lon_index.\
            extract_data(section_coords,"flood_force_merge_lon_index","integer",language)
        additional_flood_redirect_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="additional_flood_redirect_lat_index")
        extracted_data += additional_flood_redirect_lat_index.\
            extract_data(section_coords,"additional_flood_redirect_lat_index","integer",
                         language)
        additional_flood_redirect_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="additional_flood_redirect_lon_index")
        extracted_data += additional_flood_redirect_lon_index.\
            extract_data(section_coords,"additional_flood_redirect_lon_index","integer",
                         language)
        additional_flood_local_redirect = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="additional_flood_local_redirect")
        extracted_data += additional_flood_local_redirect.\
            extract_data(section_coords,"additional_flood_local_redirect","integer",
                         language)

        connection_volume_thresholds = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connection_volume_thresholds")
        extracted_data += connection_volume_thresholds.\
            extract_data(section_coords,"connection_volume_thresholds",
                         "double",language)
        connect_redirect_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_redirect_lat_index")
        extracted_data += connect_redirect_lat_index.\
            extract_data(section_coords,"connect_redirect_lat_index","integer",
                         language)
        connect_redirect_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_redirect_lon_index")
        extracted_data += connect_redirect_lon_index.\
            extract_data(section_coords,"connect_redirect_lon_index","integer",
                         language)
        connect_next_cell_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_next_cell_lat_index")
        extracted_data += connect_next_cell_lat_index.\
            extract_data(section_coords,"connect_next_cell_lat_index","integer",
                         language)
        connect_next_cell_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_next_cell_lon_index")
        extracted_data += connect_next_cell_lon_index.\
            extract_data(section_coords,"connect_next_cell_lon_index","integer",
                         language)
        connect_local_redirect = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_local_redirect")
        extracted_data += connect_local_redirect.\
            extract_data(section_coords,"connect_local_redirect","integer",
                         language)
        connect_force_merge_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_force_merge_lat_index")
        extracted_data += connect_force_merge_lat_index.\
            extract_data(section_coords,"connect_force_merge_lat_index","integer",
                         language)
        connect_force_merge_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="connect_force_merge_lon_index")
        extracted_data += connect_force_merge_lon_index.\
            extract_data(section_coords,"connect_force_merge_lon_index","integer",
                         language)
        additional_connect_redirect_lat_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="additional_connect_redirect_lat_index")
        extracted_data += additional_connect_redirect_lat_index.\
            extract_data(section_coords,"additional_connect_redirect_lat_index","integer",
                         language)
        additional_connect_redirect_lon_index = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="additional_connect_redirect_lon_index")
        extracted_data += additional_connect_redirect_lon_index.\
            extract_data(section_coords,"additional_connect_redirect_lon_index","integer",
                         language)
        additional_connect_local_redirect = \
            iodriver.advanced_field_loader(lake_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="additional_connect_local_redirect")
        extracted_data += additional_connect_local_redirect.\
            extract_data(section_coords,"additional_connect_local_redirect","integer",
                         language)
        baseflow_k = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="AGF_K")
        extracted_data += baseflow_k.\
            extract_data(coarse_section_coords,"baseflow_k","double",
                         language)
        landflow_k = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="ALF_K")
        extracted_data += landflow_k.\
            extract_data(coarse_section_coords,"landflow_k","double",
                         language)
        landflow_n = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="ALF_N")
        extracted_data += landflow_n.\
            extract_data(coarse_section_coords,"landflow_n","integer",
                         language)
        riverflow_k = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="ARF_K")
        extracted_data += riverflow_k.\
            extract_data(coarse_section_coords,"riverflow_k","double",
                         language)
        riverflow_n = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="ARF_N")
        extracted_data += riverflow_n.\
            extract_data(coarse_section_coords,"riverflow_n","integer",
                         language)
        river_directions = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="FDIR")
        extracted_data += river_directions.\
            extract_data(coarse_section_coords,"river_directions","integer",
                         language)
        landsea_mask = \
            iodriver.advanced_field_loader(hd_parameters_filepath,
                                           field_type='Generic',
                                           fieldname="FLAG")
        extracted_data += landsea_mask.\
            extract_data(coarse_section_coords,"landsea_mask","integer",
                         language)
        if write_to_text_file_filename:
            print("Writing output to {}".format(write_to_text_file_filename))
            with open(write_to_text_file_filename, "w") as output_file:
                output_file.write(extracted_data)
        else:
            print(extracted_data)

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """
    lake_data_extractor = Lake_Data_Extractor()
    lake_data_extractor.extract_data_to_run_lake_model(lake_parameters_filepath=
                                                       "/Users/thomasriddick/Documents/"
                                                       "data/HDdata/lakeparafiles/"
                                                       "lakeparas_prepare_basins_from_"
                                                       "glac1D_20210205_151552_1250.nc",
                                                       hd_parameters_filepath="/Users/"
                                                       "thomasriddick/Documents/data/"
                                                       "transient_sim_data/1/"
                                                       "hd_file_prepare_basins_from_glac1D_1250.nc",
                                                       section_coords={"min_lat":420,
                                                                       "max_lat":500,
                                                                       "min_lon":1140,
                                                                       "max_lon":1200},
                                                       language="Python",
                                                       write_to_text_file_filename="/Users/thomasriddick/Documents/"
                                                                                   "data/temp/extracted_data.txt")

if __name__ == '__main__':
    main()
