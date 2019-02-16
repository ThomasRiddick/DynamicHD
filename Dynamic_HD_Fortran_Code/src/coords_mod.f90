module coords_mod
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
end type latlon_section_coords

interface latlon_section_coords
    procedure latlat_section_coords_constructor
end interface latlon_section_coords

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
    procedure dir_based_direction_indicator_constructor
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

    pure function generic_1d_constructor(index) result(constructor)
        type(generic_1d_coords) :: constructor
        integer, intent(in) :: index
            constructor%index = index
    end function generic_1d_constructor

    pure function generic_1d_are_equal_to(this,rhs_coords) result(are_equal)
        class(generic_1d_coords), intent(in) :: this
        class(coords), intent(in) :: rhs_coords
        logical ::are_equal
            select type (rhs_coords)
            type is (generic_1d_coords)
                are_equal = (this%index == rhs_coords%index)
            end select
    end function generic_1d_are_equal_to

    function latlat_section_coords_constructor(section_min_lat,section_min_lon,&
                                               section_width_lat,section_width_lon) &
                                               result(constructor)
    type(latlon_section_coords) :: constructor
    integer :: section_min_lat
    integer :: section_min_lon
    integer :: section_width_lat
    integer :: section_width_lon
        constructor%section_min_lat = section_min_lat
        constructor%section_min_lon = section_min_lon
        constructor%section_width_lat = section_width_lat
        constructor%section_width_lon = section_width_lon
    end function latlat_section_coords_constructor

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

end module coords_mod
