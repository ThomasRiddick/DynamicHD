#include "base/field.hpp"
#include "base/section_coords.hpp"
#include "base/field_section.hpp"

class bounds {
  public:
    double west_extreme_lon;
    double east_extreme_lon;
    double north_extreme_lat;
    double south_extreme_lat;
};

class vertex_coords {};

class unstructured_grid_vertex_coords : public vertex_coords {
  public:
    unstructured_grid_vertex_coords(double* vertex_lats_in,
                                    double* vertex_lons_in);
    double* vertex_lats;
    double* vertex_lons;
};

typedef vertex_coords* vertex_coords_ptr;

class non_coincident_grid_mapper {
  public:
    field<int>* generate_cell_numbers();
    void generate_limits();
    void offset_limits(int lat_offset,
                       int lon_offset);
  protected:
    field<bool>* mask = nullptr;
    field<int>* cell_numbers = nullptr;
    field<double>* pixel_center_lats = nullptr;
    field<double>* pixel_center_lons = nullptr;
    field<vertex_coords_ptr>* cell_vertex_coords;
    section_coords* primary_area_to_consider = nullptr;
    section_coords* secondary_area_to_consider = nullptr;
    grid_params* fine_grid_params;
    grid* fine_grid = nullptr;
    grid* coarse_grid = nullptr;
    coords* coarse_cell_coords = nullptr;
    bounds* cell_bounds = nullptr;
    field<int>* section_min_lats;
    field<int>* section_min_lons;
    field<int>* section_max_lats;
    field<int>* section_max_lons;
    bool display_progress = true;
    void set_cell_numbers(int* cell_numbers_in);
    void generate_pixels_in_cell_mask(coords* cell_coords);
    void check_if_pixel_is_in_cell(coords* coords_in);
    void process_cell(coords* coords_in);
    virtual void process_pixel_for_limits(coords* coords_in) = 0;
    virtual void generate_cell_bounds() = 0;
    virtual void generate_areas_to_consider() = 0;
    virtual section_coords* generate_area_to_consider(double area_min_lon,
                                                      double area_max_lon) = 0;
    virtual bool check_if_pixel_center_is_in_bounds(double pixel_center_lat,
                                                    double pixel_center_lon) = 0;
    virtual void create_new_mask() = 0;
    virtual void assign_cell_numbers() = 0;
    virtual void print_progress() = 0;
};

class icon_icosohedral_cell_latlon_pixel_ncg_mapper :
    public non_coincident_grid_mapper {
  public:
    icon_icosohedral_cell_latlon_pixel_ncg_mapper(
      field<double>* pixel_center_lats_in,
      field<double>* pixel_center_lons_in,
      field<vertex_coords_ptr>* cell_vertex_coords_in,
      grid_params* coarse_grid_params_in,
      grid_params* fine_grid_params_in,
      bool longitudal_range_centered_on_zero_in = false);
      tuple<field<int>*,field<int>*,field<int>*,field<int>*> get_limits();
  protected:
    bool longitudal_range_centered_on_zero;
    double get_vertex_coords(coords* coords_in,
                             int vertex_num_in,
                             bool return_lat);
    void generate_areas_to_consider();
    section_coords* generate_area_to_consider(double area_min_lon,
                                              double area_max_lon);
    double calculate_line(double x,double x1,double x2,
                          double y1,double y2);
    void generate_cell_bounds();
    bool check_if_pixel_center_is_in_bounds(double pixel_center_lat,
                                            double pixel_center_lon);
    void create_new_mask();
    void assign_cell_numbers();
    void process_pixel_for_limits(coords* coords_in);
    void init_icon_icosohedral_cell_latlon_pixel_ncg_mapper();
    void print_progress();
};
