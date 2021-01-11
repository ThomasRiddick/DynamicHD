import dynamic_hd_driver
import iodriver

class Lake_Data_Extractor(Dynamic_HD_Drivers):

    def extract_data_to_run_lake_model(lake_parameters_filepath,
                                       hd_parameters_filepath,
                                       section_coords,
                                       language="Fortran"
                                       lake_initial_conditions_filepath=None,
                                       hd_initial_conditions_filepath=None):
        extracted_data = ""
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
                                           fieldname="connect_volume_thresholds")
        extracted_data += connect_volume_thresholds.\
            extract_data(section_coords,"connect_volume_thresholds",
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
        print extract_data



    # def extract_data_to_run_basin_evaluator(minima_filepath,
    #                                         raw_orography_filepath,
    #                                         cell_area_filepath,
    #                                         ):

