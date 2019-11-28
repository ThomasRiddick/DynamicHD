module map_non_coincident_grids_mod

use precision_mod
use coords_mod
use field_section_mod
use pointer_mod
use subfield_mod

implicit none

type, public :: bounds
  real(kind=double_precision) :: west_extreme_lon,east_extreme_lon
  real(kind=double_precision) :: north_extreme_lat,south_extreme_lat
end type

type, public,abstract :: vertex_coords
end type

type, public, extends(vertex_coords) :: unstructured_grid_vertex_coords
  real(kind=double_precision), dimension(:), allocatable :: vertex_lats
  real(kind=double_precision), dimension(:), allocatable :: vertex_lons
end type

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
  contains
    procedure :: generate_pixels_in_cell_mask
    procedure :: check_if_pixel_is_in_cell
    procedure :: generate_cell_numbers
    procedure :: process_cell
    procedure(generate_limits), deferred :: generate_limits
    procedure(process_pixel_for_limits), deferred :: process_pixel_for_limits
    procedure(generate_cell_bounds), deferred :: generate_cell_bounds
    procedure(generate_areas_to_consider), deferred :: generate_areas_to_consider
    procedure(check_if_pixel_center_is_in_bounds), deferred :: &
      check_if_pixel_center_is_in_bounds
    procedure(create_new_mask), deferred :: create_new_mask
    procedure(assign_cell_numbers), deferred :: assign_cell_numbers
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
    real(kind=double_precision), intent(in) :: pixel_center_lat, pixel_center_lon
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
    class(coords), intent(in) :: coords_in
  end subroutine

  subroutine generate_limits(this)
    import non_coincident_grid_mapper
    implicit none
    class(non_coincident_grid_mapper), intent(inout) :: this
  end subroutine generate_limits
end interface

type, extends(non_coincident_grid_mapper), public :: &
  icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper
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
  procedure :: generate_limits => latlon_generate_limits
end type

interface icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper
    procedure icon_icosohedral_cell_latlon_pixel_ncg_mapper_constructor
end interface icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper

contains

  subroutine generate_pixels_in_cell_mask(this,cell_coords)
    class(non_coincident_grid_mapper), intent(inout) :: this
    class(coords), pointer, intent(in) :: cell_coords
    !Despite the name this field section actually covers the entire field
      call this%create_new_mask()
      this%coarse_cell_coords => cell_coords
      call this%generate_cell_bounds()
      call this%generate_areas_to_consider()
      call this%area_to_consider_mask%for_all(check_if_pixel_is_in_cell,this)
      if (associated(this%secondary_area_to_consider_mask)) then
        call this%secondary_area_to_consider_mask%for_all(check_if_pixel_is_in_cell,this)
      end if
  end subroutine generate_pixels_in_cell_mask

  subroutine check_if_pixel_is_in_cell(this,coords_in)
    class(non_coincident_grid_mapper), intent(inout) :: this
    class(coords), pointer,intent(in) :: coords_in
    class(*), pointer :: pixel_center_lat_ptr, pixel_center_lon_ptr
    real(kind=double_precision) :: pixel_center_lat, pixel_center_lon
      pixel_center_lat_ptr => this%pixel_center_lats%get_value(coords_in)
      select type (pixel_center_lat_ptr)
        type is (real)
          pixel_center_lat = pixel_center_lat_ptr
      end select
      pixel_center_lon_ptr => this%pixel_center_lons%get_value(coords_in)
      select type (pixel_center_lon_ptr)
        type is (real)
          pixel_center_lon = pixel_center_lon_ptr
      end select
      call this%mask%set_value(coords_in,&
                               this%check_if_pixel_center_is_in_bounds(pixel_center_lat,&
                                                                       pixel_center_lon))
  end subroutine check_if_pixel_is_in_cell

  function generate_cell_numbers(this) &
      result(cell_numbers)
    class(non_coincident_grid_mapper), intent(inout) :: this
    class(field_section), pointer  :: cell_numbers
      call this%cell_vertex_coords%for_all_section(process_cell,this)
      cell_numbers =>this%cell_numbers
  end function generate_cell_numbers

  subroutine process_cell(this,coords_in)
    class(non_coincident_grid_mapper), intent(inout) :: this
    class(coords), pointer,intent(in) :: coords_in
      call this%generate_pixels_in_cell_mask(coords_in)
      call this%assign_cell_numbers()
  end subroutine process_cell

  subroutine latlon_generate_limits(this)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
      call this%cell_numbers%for_all_section(latlon_process_pixel_for_limits,this)
  end subroutine latlon_generate_limits

  subroutine latlon_process_pixel_for_limits(this,coords_in)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
    class(coords), pointer,intent(in) :: coords_in
    integer :: i,j
    integer :: working_cell_number
    class(*), pointer :: working_cell_number_ptr
      select type (coords_in)
      type is (latlon_coords)
        i = coords_in%lat
        j = coords_in%lon
      end select
      working_cell_number_ptr => this%cell_numbers%get_value(coords_in)
      select type (working_cell_number_ptr)
      type is (integer)
        working_cell_number = working_cell_number_ptr
      end select
      if (this%section_min_lats(working_cell_number) > i) then
        this%section_min_lats(working_cell_number) = i
      end if
      if (this%section_max_lats(working_cell_number) < i) then
        this%section_max_lats(working_cell_number) = i
      end if
      if (this%section_min_lons(working_cell_number) > j) then
        this%section_min_lons(working_cell_number) = j
      end if
      if (this%section_max_lons(working_cell_number) < j) then
        this%section_max_lons(working_cell_number) = j
      end if
  end subroutine latlon_process_pixel_for_limits

  subroutine latlon_pixel_generate_areas_to_consider(this)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
    real(kind=double_precision) :: rotated_west_extreme_lon, rotated_east_extreme_lon
      select type (latlon_fine_grid_shape => this%fine_grid_shape)
      type is (latlon_section_coords)
        rotated_west_extreme_lon = this%cell_bounds%west_extreme_lon - latlon_fine_grid_shape%zero_line
        rotated_east_extreme_lon = this%cell_bounds%east_extreme_lon - latlon_fine_grid_shape%zero_line
      end select
      if (rotated_west_extreme_lon < 0) then
        rotated_west_extreme_lon = rotated_west_extreme_lon + 360
      end if
      if (rotated_east_extreme_lon < 0) then
        rotated_east_extreme_lon = rotated_east_extreme_lon + 360
      end if
      if (rotated_west_extreme_lon < rotated_east_extreme_lon) then
        this%area_to_consider_mask => &
          this%latlon_pixel_generate_area_to_consider(rotated_west_extreme_lon,&
                                                      rotated_east_extreme_lon)
      else if (rotated_west_extreme_lon > rotated_east_extreme_lon) then
        this%area_to_consider_mask => &
          this%latlon_pixel_generate_area_to_consider(0.0_double_precision,rotated_east_extreme_lon)
        this%secondary_area_to_consider_mask => &
          this%latlon_pixel_generate_area_to_consider(rotated_west_extreme_lon, 360.0_double_precision)
      else
        stop "Error - cell appears to have no width"
      end if
  end subroutine  latlon_pixel_generate_areas_to_consider

  function latlon_pixel_generate_area_to_consider(this,area_min_lon, area_max_lon) &
      result(area_to_consider)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
    class(latlon_section_coords), pointer :: area_to_consider_section_coords
    class(*),pointer,dimension(:,:) :: mask
    class(subfield), pointer  :: area_to_consider
    real(kind=double_precision) :: area_min_lon, area_max_lon
    type(latlon_coords),pointer :: pixel_in_row_or_column
    class(*), pointer :: pixel_center_lat_ptr,pixel_center_lon_ptr
    real(kind=double_precision)  :: pixel_center_lat,pixel_center_lon
    integer :: area_min_lat_index, area_max_lat_index
    integer :: area_min_lon_index, area_max_lon_index
    integer :: nlat,nlon
    integer :: i
    logical :: minimum_found
      select type (latlon_fine_grid_shape => this%fine_grid_shape)
        type is (latlon_section_coords)
          nlat = latlon_fine_grid_shape%section_width_lat
          nlon = latlon_fine_grid_shape%section_width_lon
      end select
      minimum_found = .FALSE.
      allocate(pixel_in_row_or_column)
      do i=1,nlat
        pixel_in_row_or_column = latlon_coords(i,1)
        pixel_center_lat_ptr => this%pixel_center_lats%get_value(pixel_in_row_or_column)
        select type (pixel_center_lat_ptr)
          type is (real)
            pixel_center_lat = pixel_center_lat_ptr
        end select
        if ( pixel_center_lat <= this%cell_bounds%north_extreme_lat &
            .and. .not. minimum_found) then
          area_min_lat_index = i
          minimum_found = .TRUE.
        end if
        area_max_lat_index = i
        if (pixel_center_lat > &
            this%cell_bounds%south_extreme_lat) exit
      end do
      do i=1,nlon
        pixel_in_row_or_column = latlon_coords(1,i)
         pixel_center_lon_ptr => this%pixel_center_lons%get_value(pixel_in_row_or_column)
        select type (pixel_center_lon_ptr)
          type is (real)
            pixel_center_lon = pixel_center_lon_ptr
        end select
        if (pixel_center_lon <= area_min_lon .and. &
            .not. minimum_found) then
          area_min_lon_index = i
          minimum_found = .TRUE.
        end if
        area_max_lon_index = i
        if (pixel_center_lon > area_max_lon) exit
      end do
      allocate(area_to_consider_section_coords,&
               source=latlon_section_coords(area_max_lat_index, &
                                            area_min_lon_index, &
                                            area_max_lat_index - area_min_lat_index + 1, &
                                            area_max_lon_index - area_min_lon_index + 1))
      allocate(logical::mask(1:area_max_lon_index - area_min_lat_index + 1, &
                             1:area_max_lon_index - area_min_lon_index + 1))
      area_to_consider=> latlon_subfield(mask,area_to_consider_section_coords)
  end function latlon_pixel_generate_area_to_consider

  subroutine latlon_pixel_create_new_mask(this)
      class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
      class(*),pointer,dimension(:,:) :: new_mask
        select type(fine_grid_shape => this%fine_grid_shape)
        type is (latlon_section_coords)
          allocate(logical::new_mask(1:fine_grid_shape%section_width_lat, &
                                     1:fine_grid_shape%section_width_lon))
          select type(new_mask)
          type is (logical)
            new_mask = .False.
          end select
          this%mask => latlon_field_section(new_mask,fine_grid_shape)
        end select
  end subroutine latlon_pixel_create_new_mask

  subroutine icon_icosohedral_cell_generate_cell_bounds(this)
      class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
        this%cell_bounds%west_extreme_lon = &
          max(max(this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,1,.FALSE.), &
                  this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,2,.FALSE.)),&
              this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,3,.FALSE.))
        this%cell_bounds%east_extreme_lon = &
          min(min(this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,1,.FALSE.),&
                  this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,2,.FALSE.)),&
              this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,3,.FALSE.))
        this%cell_bounds%south_extreme_lat = &
          max(max(this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,1,.TRUE.),&
                this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,2,.TRUE.)),&
              this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,3,.TRUE.))
        this%cell_bounds%north_extreme_lat = &
          min(min(this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,1,.TRUE.),&
                this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,2,.TRUE.)),&
              this%icon_icosohedral_cell_get_vertex_coords(this%coarse_cell_coords,3,.TRUE.))
  end subroutine icon_icosohedral_cell_generate_cell_bounds

  function icon_icosohedral_cell_check_if_pixel_center_is_in_bounds(this,pixel_center_lat,pixel_center_lon) &
        result(is_in_bounds)
      class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
      real(kind=double_precision), intent(in) :: pixel_center_lat, pixel_center_lon
      real(kind=double_precision) :: vertex_one_lat, vertex_one_lon
      real(kind=double_precision) :: vertex_two_lat, vertex_two_lon
      real(kind=double_precision) :: vertex_three_lat, vertex_three_lon
      real(kind=double_precision) :: vertex_temp_lat, vertex_temp_lon
      logical :: is_in_bounds
      integer :: i
        select type (coarse_cell_coords => this%coarse_cell_coords)
        type is (generic_1d_coords)
          vertex_one_lat = &
            this%icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,1,.TRUE.)
          vertex_one_lon = &
            this%icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,1,.FALSE.)
          vertex_two_lat = &
            this%icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,2,.TRUE.)
          vertex_two_lon = &
            this%icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,2,.FALSE.)
          vertex_three_lat = &
            this%icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,3,.TRUE.)
          vertex_three_lon = &
            this%icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,3,.FALSE.)
        end select
        is_in_bounds = .TRUE.
        do i = 1,3
          vertex_temp_lon = vertex_three_lon
          vertex_three_lon = vertex_two_lon
          vertex_two_lon = vertex_one_lon
          vertex_one_lon = vertex_temp_lon
          vertex_temp_lat = vertex_three_lat
          vertex_three_lat = vertex_two_lat
          vertex_two_lat = vertex_one_lat
          vertex_one_lat = vertex_temp_lat
          if (vertex_one_lon-vertex_two_lon == 0) then
            is_in_bounds = is_in_bounds .and. &
                           SIGN(1.0_double_precision,pixel_center_lon - vertex_one_lon) == &
                           SIGN(1.0_double_precision,vertex_three_lon - vertex_one_lon)
          else if (vertex_two_lat - vertex_one_lat == 0) then
            is_in_bounds = is_in_bounds .and. &
                           SIGN(1.0_double_precision,pixel_center_lat - vertex_one_lat) == &
                           SIGN(1.0_double_precision,vertex_three_lat - vertex_one_lat)
          else
            is_in_bounds = is_in_bounds .and. &
                           SIGN(1.0_double_precision, &
                                this%icon_icosohedral_cell_calculate_line(pixel_center_lon, vertex_one_lon, &
                                                                          vertex_two_lon, vertex_one_lat, &
                                                                          vertex_two_lat)) == &
                           SIGN(1.0_double_precision, &
                                this%icon_icosohedral_cell_calculate_line(vertex_three_lon, vertex_one_lon, &
                                                                          vertex_two_lon, vertex_one_lat, &
                                                                          vertex_two_lat))
          end if
        end do
  end function icon_icosohedral_cell_check_if_pixel_center_is_in_bounds

  function icon_icosohedral_cell_calculate_line(x,x1,x2,y1,y2) result(y)
    real(kind=double_precision), intent(in) :: x,x1,x2,y1,y2
    real(kind=double_precision) :: y
      y = ((y2 - y1)/(x2 - x1))*(x-x1) + y1
  end function icon_icosohedral_cell_calculate_line

  function icon_icosohedral_cell_get_vertex_coords(this,coords_in,vertex_num,return_lat) result (coordinate)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(in) :: this
    class(coords), intent(in) ::  coords_in
    integer, intent(in) :: vertex_num
    logical, intent(in) :: return_lat
    real(kind=double_precision) :: coordinate
    class(*), pointer :: container_ptr
    type(container_for_pointer) :: container
      container_ptr => this%cell_vertex_coords%get_value(coords_in)
      select type (object => container%object)
       type is (unstructured_grid_vertex_coords)
        if (return_lat) then
          coordinate = object%vertex_lats(vertex_num)
        else
          coordinate = object%vertex_lons(vertex_num)
        end if
     end select
  end function icon_icosohedral_cell_get_vertex_coords

  subroutine icon_icosohedral_cell_latlon_pixel_assign_cell_numbers(this)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
      select type (mask=>this%mask)
      type is (latlon_field_section)
        select type (data=>mask%data)
        type is (logical)
          select type(cell_numbers => this%cell_numbers)
          type is (latlon_field_section)
            select type (cell_numbers_data => cell_numbers%data)
            type is (integer)
              select type (coarse_cell_coords=>this%coarse_cell_coords)
              type is (generic_1d_coords)
                  where(data) cell_numbers_data = coarse_cell_coords%index
              end select
            end select
          end select
        end select
      end select
  end subroutine icon_icosohedral_cell_latlon_pixel_assign_cell_numbers

  subroutine init_icon_icosohedral_cell_latlon_pixel_ncg_mapper(this,&
                                                                pixel_center_lats,&
                                                                pixel_center_lons,&
                                                                cell_vertex_coords,&
                                                                fine_grid_shape)
    class(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), intent(inout) :: this
    class(latlon_field_section), pointer, intent(in)  :: pixel_center_lats
    class(latlon_field_section), pointer, intent(in)  :: pixel_center_lons
    class(icon_single_index_field_section), pointer, intent(in)  :: cell_vertex_coords
    class(latlon_section_coords),pointer,intent(in) :: fine_grid_shape
    class(*),pointer,dimension(:,:) :: new_cell_numbers
      this%pixel_center_lats => pixel_center_lats
      this%pixel_center_lons => pixel_center_lons
      this%cell_vertex_coords => cell_vertex_coords
      this%fine_grid_shape => fine_grid_shape
      allocate(integer::new_cell_numbers(1:fine_grid_shape%section_width_lat, &
                                         1:fine_grid_shape%section_width_lon))
      select type(new_cell_numbers)
      type is (integer)
        new_cell_numbers = 0
      end select
      this%cell_numbers => latlon_field_section(new_cell_numbers,fine_grid_shape)
  end subroutine init_icon_icosohedral_cell_latlon_pixel_ncg_mapper

  function icon_icosohedral_cell_latlon_pixel_ncg_mapper_constructor(pixel_center_lats,&
                                                                     pixel_center_lons,&
                                                                     cell_vertex_coords,&
                                                                     fine_grid_shape) &
                                                                     result(constructor)
        type(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper), allocatable :: constructor
        class(latlon_field_section), pointer, intent(in)  :: pixel_center_lats
        class(latlon_field_section), pointer, intent(in)  :: pixel_center_lons
        class(icon_single_index_field_section), pointer, intent(in)  :: cell_vertex_coords
        class(latlon_section_coords),pointer, intent(in) :: fine_grid_shape
            allocate(constructor)
            call constructor%init_icon_icosohedral_cell_latlon_pixel_ncg_mapper(pixel_center_lats,&
                                                                                pixel_center_lons,&
                                                                                cell_vertex_coords,&
                                                                                fine_grid_shape)
  end function icon_icosohedral_cell_latlon_pixel_ncg_mapper_constructor

end module map_non_coincident_grids_mod
