type, public :: bounds
  real(kind=double_precision) :: west_extreme_lon,east_extreme_lon
  real(kind=double_precision) :: north_extreme_lat,south_extreme_lat
end type

type, public,abstract :: vertex_coords
end type

type, public, extends(vertex_coords) :: unstructured_grid_vertex_coords
  real(kind=double_precision), dimension(:), pointer :: vertex_lats
  real(kind=double_precision), dimension(:), pointer :: vertex_lons
end type

interface unstructured_grid_vertex_coords
  procedure :: unstructured_grid_vertex_coords_constructor
end interface unstructured_grid_vertex_coords

type, public, abstract :: non_coincident_grid_mapper
  class(field_section), pointer  :: mask
  class(field_section), pointer  :: cell_numbers
  class(field_section), pointer  :: pixel_center_lats
  class(field_section), pointer  :: pixel_center_lons
  class(field_section), pointer  :: cell_vertex_coords
  class(subfield), pointer  :: area_to_consider_mask
  class(subfield), pointer  :: secondary_area_to_consider_mask
  class(coords), pointer :: coarse_cell_coords
  class(bounds), pointer :: cell_bounds
  class(section_coords),pointer :: fine_grid_shape
  integer, dimension(:), pointer :: section_min_lats
  integer, dimension(:), pointer :: section_min_lons
  integer, dimension(:), pointer :: section_max_lats
  integer, dimension(:), pointer :: section_max_lons
  logical :: display_progress  = .true.
  contains
    procedure :: set_cell_numbers
    procedure :: generate_pixels_in_cell_mask
    procedure :: check_if_pixel_is_in_cell
    procedure :: generate_cell_numbers
    procedure :: process_cell
    procedure :: generate_limits
    procedure :: offset_limits
    procedure(process_pixel_for_limits), deferred :: process_pixel_for_limits
    procedure(generate_cell_bounds), deferred :: generate_cell_bounds
    procedure(generate_areas_to_consider), deferred :: generate_areas_to_consider
    procedure(check_if_pixel_center_is_in_bounds), deferred :: &
      check_if_pixel_center_is_in_bounds
    procedure(create_new_mask), deferred :: create_new_mask
    procedure(assign_cell_numbers), deferred :: assign_cell_numbers
    procedure(print_progress), deferred :: print_progress
end type

abstract interface
  subroutine generate_cell_bounds(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
  end subroutine generate_cell_bounds

  subroutine generate_areas_to_consider(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
  end subroutine generate_areas_to_consider

  function check_if_pixel_center_is_in_bounds(this,pixel_center_lat,pixel_center_lon) &
      result(is_in_bounds)
    import non_coincident_grid_mapper
    import double_precision
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
    real(kind=double_precision), intent(inout) :: pixel_center_lat, pixel_center_lon
    logical :: is_in_bounds
  end function check_if_pixel_center_is_in_bounds

  subroutine create_new_mask(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
  end subroutine create_new_mask

  subroutine assign_cell_numbers(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
  end subroutine

  subroutine process_pixel_for_limits(this,coords_in)
    import non_coincident_grid_mapper
    import coords
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
    class(coords), pointer, intent(inout) :: coords_in
  end subroutine

  subroutine print_progress(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(in) :: this
  end subroutine print_progress
end interface

type, extends(non_coincident_grid_mapper), public :: &
  icon_icosohedral_cell_latlon_pixel_ncg_mapper
  logical :: longitudal_range_centered_on_zero
contains
  procedure :: icon_icosohedral_cell_get_vertex_coords
  procedure :: latlon_pixel_generate_area_to_consider
  procedure, nopass :: icon_icosohedral_cell_calculate_line
  procedure :: generate_cell_bounds => icon_icosohedral_cell_generate_cell_bounds
  procedure :: generate_areas_to_consider => latlon_pixel_generate_areas_to_consider
  procedure :: check_if_pixel_center_is_in_bounds => &
    icon_icosohedral_cell_check_if_pixel_center_is_in_bounds
  procedure :: create_new_mask => latlon_pixel_create_new_mask
  procedure :: assign_cell_numbers => &
    icon_icosohedral_cell_latlon_pixel_assign_cell_numbers
  procedure :: process_pixel_for_limits => latlon_process_pixel_for_limits
  procedure :: init_icon_icosohedral_cell_latlon_pixel_ncg_mapper
  procedure :: icon_icosohedral_cell_latlon_pixel_ncg_mapper_destructor
  procedure :: print_progress => icon_icosohedral_cell_print_progress
end type

interface icon_icosohedral_cell_latlon_pixel_ncg_mapper
    procedure icon_icosohedral_cell_latlon_pixel_ncg_mapper_constructor
end interface icon_icosohedral_cell_latlon_pixel
