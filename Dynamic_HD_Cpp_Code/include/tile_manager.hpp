#ifndef INCLUDE_TILE_MANAGER_
#define INCLUDE_TILE_MANAGER_

using namespace std;

template <typename field_type>
class tile_manager{
  public:
    ~tile_manager();
    virtual void setup_tile_manager_from_field(field<field_type>* field_in,
                                               int number_of_tiles_in,
                                               int scale_factor = 1);
    virtual void setup_tile_manager_from_files(tile_file_directory_filename_in);
    virtual field<field_type>* get_tile(int tile_number_in);
  protected:
    grid* _grid = nullptr;
    field<field_type>* data_field = nullptr;
    TileFileDirectory* tile_file_directory = nullptr;
    int number_of_tiles = 0;
    int scale_factor = 0;
    bool full_field_available = false;
}

template <typename field_type>
class latlon_tile_manager : tile_manager<field_type>{
  public:
    ~latlon_tile_manager();
    void setup_tile_manager_from_field(field<field_type>* field_in,
                                       int number_of_tiles,
                                       int scale_factor);
    void setup_tile_manager_from_files(tile_file_directory_filename);
    field<field_type>* get_tile(int tile_number);
  private:
    int nlat_tiles;
    int nlon_tiles;
    int nlat_coarse
    int nlon_coarse
    int nlat
    int nlon
}

template <typename field_type>
void latlon_tile_manager<field_type>::setup_tile_manager_from_field(field<field_type>* field_in,
                                                                    int number_of_tiles_in,
                                                                    int scale_factor = 1){
  nlat_tiles = floor(sqrt(number_of_tiles_in/2.0));
  if (nlat_tiles == 0){
    nlat_tiles = 1;
  }
  nlon_tiles = nlat_tiles*2;
  if (nlon_tiles == 0){
    nlon_tiles = 1
  }
  number_of_tiles = nlat_tiles*nlon_tiles
  full_field_available = true
}

template <typename field_type>
field<field_type>* latlon_tile_manager<field_type>::get_tile(int tile_number){
  if (full_field_available){
    tile_column = tile_number/nlat_tiles
    tile_row = tile_number - tile_column*nlat_tiles
    min_i = tile_row*(nlat_coarse/nlat_tiles)*scale_factor
    if (tile_row != (nlat_tiles-1)){
      max_i = ((tile_row+1)*(nlat_coarse/nlat_tiles)*scale_factor)-1
    } else {
      max_i = (nlat_coarse*scale_factor)-1
    }
    min_j = tile_column*(nlon_coarse/nlon_tiles)*scale_factor
    if (tile_column != (nlon_tiles-1)){
      max_j = ((tile_column+1)*(nlon_coarse/nlon_tiles)*scale_factor)-1
    else {
      max_j = (nlon_coarse*scale_factor) - 1
    }
    return data_field.get_field_section(min_i,min_j,max_i,max_j);
  } else {
    iterate over tile file
    if (tile_file_entry.check_for_overlap(working_tile_min_lat,
                                          working_tile_max_lat,
                                          working_tile_min_lon,
                                          working_tile_max_lon)){
      tile_patches_list = tile_file_entry.load_patch(working_tile_min_lat,
                                                     working_tile_max_lat,
                                                     working_tile_min_lon,
                                                     working_tile_max_lon)
    }
    for each entry in tile_patches list{
      output_field.set_patch()
    }
    return output_field
  }
}

#endif /* define INCLUDE_TILE_MANAGER_

