module coords_mod
use unstructured_grid_mod
use precision_mod
implicit none

!> An abstract class for holding coordinates on a generic grid
type, abstract :: coords
contains
    !> A function to check if this coords object is the same coordinates
    !! object passed into it; if this is the case return TRUE else return
    !! FALSE
    procedure(are_equal_to), deferred :: are_equal_to
end type coords

abstract interface
    pure function are_equal_to(this,rhs_coords) result(are_equal)
        import coords
        implicit none
        class(coords), intent(in) :: this
        class(coords), intent(in) :: rhs_coords
        logical ::are_equal
    end function are_equal_to
end interface

!> A concrete subclass of coords implementing latitude longitude coordinates
type, extends(coords) :: latlon_coords
    !> Latitude
    integer :: lat
    !> Longitude
    integer :: lon
    contains
        !> A function to check if this coords object is the same coordinates
        !! object passed into it; if this is the case return TRUE else return
        !! FALSE
        procedure :: are_equal_to => latlon_are_equal_to
end type latlon_coords

interface latlon_coords
    procedure latlon_coords_constructor
end interface latlon_coords

!> A concrete subclass of coords implementing generic 1d coordinates
type, extends(coords) ::  generic_1d_coords
    !> Single index
    integer :: index
    logical :: use_internal_index
    contains
        !> A function to check if this coords object is the same coordinates
        !! object passed into it; if this is the case return TRUE else return
        !! FALSE
        procedure :: are_equal_to => generic_1d_are_equal_to
end type generic_1d_coords

interface generic_1d_coords
    procedure generic_1d_constructor
end interface generic_1d_coords

!> An abstract class for holding the coordinates of a subsection of a generic
!! grid
type, abstract :: section_coords
end type section_coords

!> A concrete subclass of section coords for holding sections of a latitude
!! logitude grid
type, extends(section_coords) :: latlon_section_coords
    !> The minimum latitude of the section
    integer :: section_min_lat
    !> The minimum longitude of the section
    integer :: section_min_lon
    !> The latitudinal width of the section
    integer :: section_width_lat
    !> The longitudinal width of the section
    integer :: section_width_lon
    !> The actual longitude of the zero line (i.e. vertical edge of the array)
    real(kind=double_precision) :: zero_line
end type latlon_section_coords

interface latlon_section_coords
    procedure latlon_section_coords_constructor
end interface latlon_section_coords

type, extends(latlon_section_coords) :: irregular_latlon_section_coords
    integer :: cell_number
    integer, dimension(:), pointer :: list_of_cell_numbers
    integer, dimension(:,:), pointer :: cell_numbers
    integer, dimension(:), pointer :: section_min_lats
    integer, dimension(:), pointer :: section_min_lons
    integer, dimension(:), pointer :: section_max_lats
    integer, dimension(:), pointer :: section_max_lons
    contains
        procedure :: irregular_latlon_section_coords_destructor
end type irregular_latlon_section_coords

public :: irregular_latlon_section_coords_constructor, &
                 multicell_irregular_latlon_section_coords_constructor

interface irregular_latlon_section_coords
    procedure :: irregular_latlon_section_coords_constructor, &
                 multicell_irregular_latlon_section_coords_constructor
end interface irregular_latlon_section_coords

type, extends(section_coords) :: generic_1d_section_coords
    integer, dimension(:,:), pointer :: cell_neighbors
    integer, dimension(:,:), pointer :: cell_secondary_neighbors
    integer, dimension(:), pointer :: edge_cells
    integer, dimension(:), pointer :: subfield_indices
    integer, dimension(:), pointer :: full_field_indices
    logical, dimension(:), pointer :: mask
    class(unstructured_grid), pointer :: grid
    contains
        procedure :: get_cell_neighbors
        procedure :: get_cell_secondary_neighbors
        procedure :: get_section_mask
        procedure :: get_edge_cells
        procedure :: get_subfield_indices
        procedure :: get_full_field_indices
        procedure :: generic_1d_section_coords_destructor
end type generic_1d_section_coords

interface generic_1d_section_coords
    procedure :: generic_1d_section_coords_constructor
end interface generic_1d_section_coords

!> An abstract class holding a generic indicator of cell that
!! a given cell flows to; this could be implemented as a direction
!! or as the coordinates/index of the next cell
type, abstract :: direction_indicator
contains
    !> Check if this direction indicator is equal to a given integer (works
    !! only if it can be compared to a single integer); if it is return TRUE
    !! else return FALSE
    procedure :: is_equal_to_integer
    !> Check if this direction indicator object is equal to  a direction
    !! given as a polymorphic pointer to an object of the correct type;
    !! if it is return TRUE else return FALSE
    procedure(is_equal_to), deferred :: is_equal_to
end type direction_indicator

abstract interface
    pure function is_equal_to(this,value) result(is_equal)
        import direction_indicator
        class(direction_indicator), intent(in) :: this
        class(*), intent(in), pointer :: value
        logical :: is_equal
    end function is_equal_to
end interface

!> A concrete subclass of direction indicator which holds a number based
!! direction indicator - for example for a latitude longitude grid this
!! is often a number between 1 and 9 indicating direction according to
!! the direction from the centre of a numeric keyboard (the D9 method)
!! with 5 as a sink. However any other single number system is also
!! possible
type, extends(direction_indicator) :: dir_based_direction_indicator
    integer :: direction
    contains
        !> Getter for the direction indicator integer value
        procedure :: get_direction => dir_based_direction_indicator_get_direction
        !> Check if this direction indicator object is equal to a given value
        !! where the value is supplied as a polymorphic pointer to an integer
        !! if it is return TRUE else return FALSE
        procedure :: is_equal_to => dir_based_direction_indicator_is_equal_to
end type dir_based_direction_indicator

interface dir_based_direction_indicator
    procedure :: dir_based_direction_indicator_constructor
end interface

type, extends(direction_indicator) :: index_based_direction_indicator
    integer :: index
    contains
        !> Getter for the direction indicator integer value
        procedure :: get_direction => index_based_direction_indicator_get_direction
        !> Check if this direction indicator object is equal to a given value
        !! where the value is supplied as a polymorphic pointer to an integer
        !! if it is return TRUE else return FALSE
        procedure :: is_equal_to => index_based_direction_indicator_is_equal_to
end type index_based_direction_indicator

interface index_based_direction_indicator
    procedure index_based_direction_indicator_constructor
end interface

contains

    function is_equal_to_integer(this,value_in) result(is_equal)
        class(direction_indicator), intent(in) :: this
        integer :: value_in
        class(*), pointer :: value
        logical :: is_equal
            allocate(value,source=value_in)
            is_equal = this%is_equal_to(value)
            deallocate(value)
    end function is_equal_to_integer

    pure function latlon_coords_constructor(lat,lon) result(constructor)
        type(latlon_coords) :: constructor
        integer, intent(in) :: lat
        integer, intent(in) :: lon
            constructor%lat = lat
            constructor%lon = lon
    end function latlon_coords_constructor

    pure function latlon_are_equal_to(this,rhs_coords) result(are_equal)
        class(latlon_coords), intent(in) :: this
        class(coords), intent(in) :: rhs_coords
        logical ::are_equal
            select type (rhs_coords)
            type is (latlon_coords)
                are_equal = (this%lat == rhs_coords%lat) .and. &
                            (this%lon == rhs_coords%lon)
            end select
    end function latlon_are_equal_to

    pure function generic_1d_constructor(index,use_internal_index_in) result(constructor)
        type(generic_1d_coords) :: constructor
        integer, intent(in) :: index
        logical, intent(in), optional :: use_internal_index_in
        logical :: use_internal_index
            if (present(use_internal_index_in)) then
                use_internal_index = use_internal_index_in
            else
                use_internal_index = .false.
            end if
            constructor%index = index
            constructor%use_internal_index = use_internal_index
    end function generic_1d_constructor

    pure function generic_1d_are_equal_to(this,rhs_coords) result(are_equal)
        class(generic_1d_coords), intent(in) :: this
        class(coords), intent(in) :: rhs_coords
        logical ::are_equal
            select type (rhs_coords)
            type is (generic_1d_coords)
                !Be careful of operator precedence here
                are_equal = ((this%index == rhs_coords%index) .and. &
                             (this%use_internal_index .eqv. &
                             rhs_coords%use_internal_index))
            end select
    end function generic_1d_are_equal_to

    function latlon_section_coords_constructor(section_min_lat,section_min_lon,&
                                               section_width_lat,section_width_lon,&
                                               zero_line) &
                                               result(constructor)
    type(latlon_section_coords) :: constructor
    integer :: section_min_lat
    integer :: section_min_lon
    integer :: section_width_lat
    integer :: section_width_lon
    real(kind=double_precision),optional :: zero_line
        constructor%section_min_lat = section_min_lat
        constructor%section_min_lon = section_min_lon
        constructor%section_width_lat = section_width_lat
        constructor%section_width_lon = section_width_lon
        if(present(zero_line)) then
            constructor%zero_line = zero_line
        else
            constructor%zero_line = 0.0
        end if
    end function latlon_section_coords_constructor

    function irregular_latlon_section_coords_constructor(cell_number_in,cell_numbers_in, &
                                                         section_min_lats_in, &
                                                         section_min_lons_in, &
                                                         section_max_lats_in, &
                                                         section_max_lons_in, &
                                                         lat_offset_in) &
                                                          result(constructor)
    type(irregular_latlon_section_coords) :: constructor
    integer, dimension(:,:), pointer :: cell_numbers_in
    integer, dimension(:), pointer :: section_min_lats_in
    integer, dimension(:), pointer :: section_min_lons_in
    integer, dimension(:), pointer :: section_max_lats_in
    integer, dimension(:), pointer :: section_max_lons_in
    integer :: cell_number_in
    integer, optional :: lat_offset_in
    integer lat_offset
        if (present(lat_offset_in)) then
            lat_offset = lat_offset_in
        else
            lat_offset = 0
        end if
        constructor%section_max_lons => section_max_lons_in
        constructor%section_min_lons => section_min_lons_in
        allocate(constructor%section_max_lats(size(section_max_lats_in)))
        allocate(constructor%section_min_lats(size(section_min_lats_in)))
        constructor%section_max_lats(:) = section_max_lats_in(:) + lat_offset
        constructor%section_min_lats(:) = section_min_lats_in(:) + lat_offset
        constructor%cell_numbers => cell_numbers_in
        constructor%list_of_cell_numbers => null()
        constructor%cell_number = cell_number_in
        constructor%section_min_lat = constructor%section_min_lats(cell_number_in)
        constructor%section_min_lon = constructor%section_min_lons(cell_number_in)
        constructor%section_width_lat = &
            constructor%section_max_lats(cell_number_in) + 1 - &
                constructor%section_min_lats(cell_number_in)
        constructor%section_width_lon = &
            constructor%section_max_lons(cell_number_in) + 1 - &
                constructor%section_min_lons(cell_number_in)
        constructor%zero_line = 0.0
    end function irregular_latlon_section_coords_constructor

    subroutine irregular_latlon_section_coords_destructor(this)
        class(irregular_latlon_section_coords) :: this
            deallocate(this%section_max_lats)
            deallocate(this%section_min_lats)
    end subroutine irregular_latlon_section_coords_destructor

function multicell_irregular_latlon_section_coords_constructor(list_of_cell_numbers_in,cell_numbers_in, &
                                                               section_min_lats_in, &
                                                               section_min_lons_in, &
                                                               section_max_lats_in, &
                                                               section_max_lons_in) &
                                                                 result(constructor)
    type(irregular_latlon_section_coords) :: constructor
    integer, dimension(:,:), pointer :: cell_numbers_in
    integer, dimension(:), pointer :: section_min_lats_in
    integer, dimension(:), pointer :: section_min_lons_in
    integer, dimension(:), pointer :: section_max_lats_in
    integer, dimension(:), pointer :: section_max_lons_in
    integer, dimension(:), pointer :: list_of_cell_numbers_in
    integer :: section_min_lat
    integer :: section_min_lon
    integer :: section_max_lat
    integer :: section_max_lon
    integer :: i
    integer :: working_cell_number
        constructor%section_min_lons => section_min_lons_in
        constructor%section_max_lons => section_max_lons_in
        allocate(constructor%section_min_lats(size(section_min_lats_in)))
        allocate(constructor%section_max_lats(size(section_max_lats_in)))
        constructor%section_min_lats(:) = section_min_lats_in(:)
        constructor%section_max_lats(:) = section_max_lats_in(:)
        constructor%cell_numbers => cell_numbers_in
        constructor%cell_number = 0
        constructor%list_of_cell_numbers => list_of_cell_numbers_in
        working_cell_number = list_of_cell_numbers_in(1)
        section_min_lat=section_min_lats_in(working_cell_number)
        section_min_lon=section_min_lons_in(working_cell_number)
        section_max_lat=section_max_lats_in(working_cell_number)
        section_max_lon=section_max_lons_in(working_cell_number)
        do  i = 2,size(list_of_cell_numbers_in)
            working_cell_number = list_of_cell_numbers_in(i)
            section_min_lat=min(section_min_lats_in(working_cell_number), &
                                section_min_lat)
            section_min_lon=min(section_min_lons_in(working_cell_number), &
                                section_min_lon)
            section_max_lat=max(section_max_lats_in(working_cell_number), &
                                section_max_lat)
            section_max_lon=max(section_max_lons_in(working_cell_number), &
                                section_max_lon)
        end do
        constructor%section_min_lat = section_min_lat
        constructor%section_min_lon = section_min_lon
        constructor%section_width_lat = &
            section_max_lat + 1 - section_min_lat
        constructor%section_width_lon = &
            section_max_lon + 1 - section_min_lon
        constructor%zero_line = 0.0
        constructor%zero_line = 0.0
    end function multicell_irregular_latlon_section_coords_constructor

    function generic_1d_section_coords_constructor(cell_neighbors_in,&
                                                   cell_secondary_neighbors_in,&
                                                   mask_in) &
                                                    result(constructor)
        integer, dimension(:,:), pointer, intent(in) :: cell_neighbors_in
        integer, dimension(:,:), pointer, intent(in) :: cell_secondary_neighbors_in
        logical, dimension(:), pointer, intent(in), optional :: mask_in
        type(generic_1d_section_coords) :: constructor
            constructor%cell_neighbors => cell_neighbors_in
            constructor%cell_secondary_neighbors => cell_secondary_neighbors_in
            constructor%edge_cells => null()
            constructor%subfield_indices => null()
            constructor%full_field_indices => null()
            if (present(mask_in)) then
                constructor%mask => mask_in
            else
                allocate(constructor%mask(size(cell_neighbors_in,1)))
                constructor%mask = .True.
            end if
    end function

    subroutine generic_1d_section_coords_destructor(this)
        class(generic_1d_section_coords), intent(inout) :: this
            deallocate(this%mask)
    end subroutine generic_1d_section_coords_destructor

    function get_cell_neighbors(this) result(cell_neighbors)
        class(generic_1d_section_coords), intent(inout) :: this
        integer, dimension(:,:), pointer :: cell_neighbors
            if (associated(this%cell_neighbors)) then
                cell_neighbors=>this%cell_neighbors
            else
                cell_neighbors => &
                    this%grid%generate_cell_neighbors(this%mask, &
                                                      this%get_subfield_indices(), &
                                                      this%get_full_field_indices())
            end if
    end function get_cell_neighbors

    function get_section_mask(this) result(mask)
        class(generic_1d_section_coords), intent(inout) :: this
        logical, dimension(:), pointer :: mask
            mask => this%mask
    end function get_section_mask

    function get_cell_secondary_neighbors(this) result(cell_secondary_neighbors)
        class(generic_1d_section_coords), intent(inout) :: this
        integer, dimension(:,:), pointer :: cell_secondary_neighbors
            if (associated(this%cell_secondary_neighbors)) then
                cell_secondary_neighbors => this%cell_secondary_neighbors
            else
                cell_secondary_neighbors => &
                    this%grid%generate_cell_secondary_neighbors(this%get_subfield_indices(), &
                                                                this%get_full_field_indices(), &
                                                                this%get_cell_neighbors())
            end if
    end function get_cell_secondary_neighbors

    function get_edge_cells(this) result(edge_cells)
        class(generic_1d_section_coords), intent(inout) :: this
        integer, dimension(:), pointer :: edge_cells
            if (associated(this%edge_cells)) then
                edge_cells => this%edge_cells
            else
                edge_cells => &
                    this%grid%generate_edge_cells(this%get_cell_neighbors(), &
                                                  this%get_cell_secondary_neighbors())
            end if
    end function get_edge_cells

    function get_subfield_indices(this) result(subfield_indices)
        class(generic_1d_section_coords), intent(inout) :: this
        integer, dimension(:), pointer :: subfield_indices
            if (associated(this%subfield_indices)) then
                subfield_indices => this%subfield_indices
            else
                subfield_indices => &
                    this%grid%generate_subfield_indices(this%mask)
            end if
    end function get_subfield_indices

    function get_full_field_indices(this) result(full_field_indices)
        class(generic_1d_section_coords), intent(inout) :: this
        integer, dimension(:), pointer :: full_field_indices
            if (associated(this%subfield_indices)) then
                full_field_indices => this%full_field_indices
            else
                full_field_indices => &
                    this%grid%generate_full_field_indices(this%mask,&
                                                          this%get_subfield_indices())
            end if
    end function get_full_field_indices

    pure function dir_based_direction_indicator_constructor(direction) result(constructor)
        type(dir_based_direction_indicator) :: constructor
        integer, intent(in) :: direction
            constructor%direction = direction
    end function dir_based_direction_indicator_constructor

    pure function dir_based_direction_indicator_get_direction(this) result(direction)
        class(dir_based_direction_indicator), intent(in) :: this
        integer :: direction
            direction = this%direction
    end function dir_based_direction_indicator_get_direction

    pure function dir_based_direction_indicator_is_equal_to(this,value) result(is_equal)
        class(dir_based_direction_indicator), intent(in) :: this
        class(*), pointer, intent(in) :: value
        logical :: is_equal
            select type(value)
                type is (integer)
                if (this%direction == value) then
                    is_equal = .True.
                else
                    is_equal = .False.
                end if
            end select
    end function dir_based_direction_indicator_is_equal_to

    pure function index_based_direction_indicator_constructor(index) result(constructor)
        type(index_based_direction_indicator) :: constructor
        integer, intent(in) :: index
            constructor%index = index
    end function index_based_direction_indicator_constructor

    pure function index_based_direction_indicator_get_direction(this) result(direction)
        class(index_based_direction_indicator), intent(in) :: this
        integer :: direction
            direction = this%index
    end function index_based_direction_indicator_get_direction

    pure function index_based_direction_indicator_is_equal_to(this,value) result(is_equal)
        class(index_based_direction_indicator), intent(in) :: this
        class(*), pointer, intent(in) :: value
        logical :: is_equal
            select type(value)
                type is (integer)
                if (this%index == value) then
                    is_equal = .True.
                else
                    is_equal = .False.
                end if
            end select
    end function index_based_direction_indicator_is_equal_to

end module coords_mod
