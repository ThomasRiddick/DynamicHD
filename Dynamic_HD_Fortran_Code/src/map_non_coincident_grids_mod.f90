module map_non_coincident_grids

implicit none

type, public, abstract :: non_coincident_grid_mapper
  class(field_section), pointer  :: mask
  class(field_section), pointer  :: pixel_center_lats
  class(field_section), pointer  :: pixel_center_lons
  class(field_section), pointer  :: area_to_consider_mask
  class(field_section), pointer  :: secondary_area_to_consider_mask
  class(bounds), pointer :: cell_bounds
  class(section_coords) :: fine_grid_shape
  contains
    procedure :: generate_pixels_in_cell_mask
    procedure :: check_if_pixel_is_in_cell
    procedure(generate_cell_bounds), deferred, nopass :: generate_cell_bounds
    procedure(generate_area_to_consider), deferred, nopass :: generate_area_to_consider
    procedure(check_if_pixel_center_is_in_bounds), deferred, nopass :: &
      check_if_pixel_center_is_in_bounds
end type

abstract interface
  subroutine generate_cell_bounds(this,cell_coords)
    import non_coincident_grid_mapper
    import coords
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
    class(coords), intent(in) :: cell_coords
  end subroutine generate_cell_bounds

  subroutine generate_area_to_consider(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
  end subroutine generate_area_to_consider

  subroutine check_if_pixel_center_is_in_bounds(this,pixel_center_lat,pixel_center_lon)
    import non_coincident_grid_mapper
    import double_precision
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
    real(kind=double_precision) :: pixel_center_lat, pixel_center_lon
  end subroutine check_if_pixel_center_is_in_bounds
end interface

type, extendeds(non_coincident_grid_mapper), public ::
  unstructured_cell_latlon_pixel_non_coincident_grid_mapper
contains
  procedure, nopass :: generate_cell_bounds => unstructured_cell_generate_cell_bounds
  procedure, nopass :: generate_area_to_consider => latlon_pixel_generate_area_to_consider
  procedure, nopass :: check_if_pixel_center_is_in_bounds = >
    unstructured_cell_check_if_pixel_center_is_in_bounds
end type

function generate_pixels_in_cell_mask(this,cell_coords) result(mask)
  class(non_coincident_grid_mapper), intent(inout) :: this
  class(coords), intent(in) :: cell_coords
  !Despite the name this field section actually covers the entire field
  class(field_section), pointer  :: mask
  !Despite the name these section coords actually cover the entire field
    call this%generate_cell_bounds(this,cell_coords)
    call this%generate_area_to_consider(this)
    call this%area_to_consider%for_all(check_if_pixel_is_in_cell,this)
    if (allocated(this%secondary_area_to_consider)) then
      call this%secondary_area_to_consider%for_all(check_if_pixel_is_in_cell,this)
    end if
    mask => this%mask
end

subroutine check_if_pixel_is_in_cell(this,coords_in)
  class(non_coincident_grid_mapper), intent(inout) :: this
  class(coords), pointer :: coords_in
  real(kind=double_precision) :: pixel_center_lat, pixel_center_lon
    pixel_center_lat = this%pixel_center_lats(coords_in)
    pixel_center_lon = this%pixel_center_lons(coords_in)
    this%mask(coords_in) = check_if_pixel_center_is_in_bounds(pixel_center_lat,&
                                                              pixel_center_lon)
end function check_if_pixel_is_in_cell

subroutine latlon_pixel_generate_area_to_consider(this)
  class(non_coincident_grid_mapper), intent(inout) :: this
  real(kind=double_precision) :: rotated_west_extreme_lon, rotated_east_extreme_lon
  rotated_west_extreme_lon = cell_bounds%west_extreme_lon - fine_grid_shape%zero_line
  rotated_east_extreme_lon = cell_bounds%east_extreme_lon - fine_grid_shape%zero_line
  if (rotated_west_extreme_lon < 0) then
    rotated_west_extreme_lon = rotated_west_extreme_lon + 360.0
  end
  if (rotated_east_extreme_lon < 0)
    rotated_east_extreme_lon = rotated_east_extreme_lon + 360.0
  end
  if (rotated_west_extreme_lon < rotated_east_extreme_lon) then
    area_min_lat
    area_max_lat
  else if (rotated_west_extreme_lon < rotated_east_extreme_lon) then
    latlon_section_coords(fine_grid_shape%section_min_lat,
                          area_min_lat,
                          fine_grid_shape%section_width_lat,
                          area_max_lat - area_min_lat + 1)
  else
    stop "Error - cell appears to have no width"
  end
    latlon_section_coords(fine_grid_shape%section_min_lat,
                          area_min_lat,
                          fine_grid_shape%section_width_lat,
                          area_max_lat - area_min_lat + 1)
    CREATE SUBFIELDS
end subroutine  latlon_pixel_generate_area_to_consider

end module map_non_coincident_grids
