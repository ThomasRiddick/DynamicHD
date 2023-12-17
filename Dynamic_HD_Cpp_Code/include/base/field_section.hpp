INHERIT FROM FIELD!!!
use coords_mod
implicit none
private
public :: latlon_field_section_constructor

!> A class to manipulate a limited section of a given field of data using coordinates
!! that encompass the entire field of data. Actually can generally manipulate the entire
!! field of data; it simply stores the boundaries of the section.
type, public, abstract :: field_section
    contains
    !> Wrapper to a set an integer value at given set of coordinates
    procedure :: set_integer_value
    !> Wrapper to a set a real value at given set of coordinates
    procedure :: set_real_value
    !> Wrapper to a set a logical value at given set of coordinates
    procedure :: set_logical_value
    !> Generic type bound procedure to set a given value at a given coordinate
    !! this then calls one of the type specific wrappers for setting values
    generic :: set_value => set_integer_value, set_real_value, set_logical_value
    !> Get an unlimited polymorphic pointer to a value at the given coordinates
    procedure(get_value), deferred :: get_value
    !> Given a pointer to an unlimited polymorphic variable set it at the given
    !! coordinates
    procedure(set_generic_value), deferred :: set_generic_value
    !> For each cell in the section
    procedure(for_all_section), deferred :: for_all_section
    !> Print the entire data field
    procedure(print_field_section), deferred :: print_field_section
    !> Deallocate the data object
    procedure(deallocate_data), deferred :: deallocate_data
end type field_section

abstract interface
    pure function get_value(this,coords_in) result(value)
        import field_section
        import coords
        implicit none
        class(field_section), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
    end function get_value

    subroutine set_generic_value(this,coords_in,value)
        import field_section
        import coords
        implicit none
        class(field_section) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer, intent(in) :: value
    end subroutine

    subroutine for_all_section(this,subroutine_in,calling_object)
        import field_section
        implicit none
        class(field_section) :: this
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords), pointer, intent(inout) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        class(*), intent(inout) :: calling_object
    end subroutine for_all_section

    subroutine print_field_section(this)
        import field_section
        implicit none
        class(field_section) :: this
    end subroutine print_field_section

    subroutine deallocate_data(this)
        import field_section
        implicit none
        class(field_section) :: this
    end subroutine deallocate_data
end interface

!> A concrete subclass of field section for a latitude longitude grid
type, extends(field_section), public :: latlon_field_section
    !> A pointer to 2D array to hold the latitude longitude field section data;
    !! in fact a pointer to the entire field both point in and outside the section
    class(*), dimension (:,:), pointer :: data
    !> Number of latitudinal points in the field
    integer                  :: nlat
    !> Number of longitudinal points in the field
    integer                  :: nlon
    !> Minimum latitude of the section of field that is of interest
    integer                  :: section_min_lat
    !> Minimum longitude of the section of field that is of interest
    integer                  :: section_min_lon
    !> Maximum latitude of the section of field that is of interest
    integer                  :: section_max_lat
    !> Maximum longitude of the section of field that is of interest
    integer                  :: section_max_lon
    !> Wrap the field east west or not
    logical                  :: wrap
    contains
        private
        !> Initialize this latitude longitude field section. Arguments are a pointer to
        !! the input data array and the section coords of the section of the field that
        !! is of interest
        procedure :: init_latlon_field_section
        !> Set the data field to a given polymorphic value at a given latitude and longitude
        !! with no consideration for wrapping
        procedure :: set_data_array_element => latlon_set_data_array_element
        !> Return the number of the latitude points in the entire grid
        procedure, public :: get_nlat
        !> Return the number of the longitude points in the entire grid
        procedure, public :: get_nlon
        !> Return the value of the wrap flag
        procedure, public :: get_wrap
        !> Returns a pointer to a copy of the array of data (this copy is created by
        !! this function using sourced allocation)
        procedure, public :: get_data => latlon_get_data
        !> Getter for section minimum latitude
        procedure, public :: get_section_min_lat
        !> Getter for section minimum longitude
        procedure, public :: get_section_min_lon
        !> Getter for section maximum latitude
        procedure, public :: get_section_max_lat
        !> Getter for section maximum longitude
        procedure, public :: get_section_max_lon
        !> Given a pointer to an unlimited polymorphic variable set it at the given
        !! latitude logitude coordinates
        procedure, public :: set_generic_value => latlon_set_generic_value
        !> Return an unlimited polymorphic pointer to a value at the given latitude
        !! longitude coordinates
        procedure, public :: get_value => latlon_get_value
        !> For each cell in the section
        procedure, public :: for_all_section => latlon_for_all_section
        !> Print the entire latitude longitude data field
        procedure, public :: print_field_section => latlon_print_field_section
        !> Deallocate the latitude longitude data array
        procedure, public :: deallocate_data => latlon_deallocate_data
end type latlon_field_section

interface latlon_field_section
    procedure latlon_field_section_constructor
end interface latlon_field_section

!> A concrete subclass of field section for an  icon single index grid
type, extends(field_section), public :: icon_single_index_field_section
    !> A pointer to 2D array to hold the icon single index field section data;
    !! in fact a pointer to the entire field both point in and outside the section
    class(*), dimension (:), pointer :: data
    !> A mask with false outside the section and true inside
    logical, dimension(:), pointer :: mask
    !> Number of points in the field
    integer                  :: num_points
    integer, dimension(:,:), pointer :: cell_neighbors
    integer, dimension(:,:), pointer :: cell_secondary_neighbors
    contains
        private
        !> Initialize this latitude longitude field section. Arguments are a pointer to
        !! the input data array and the section coords of the section of the field that
        !! is of interest
        procedure :: init_icon_single_index_field_section
        !> Set the data field to a given polymorphic value at a given latitude and longitude
        !! with no consideration for wrapping
        procedure :: set_data_array_element => icon_single_index_set_data_array_element
        procedure, public :: get_mask
        !> Given a pointer to an unlimited polymorphic variable set it at the given
        !!  coordinates
        procedure, public :: set_generic_value => icon_single_index_set_generic_value
        !> Return an unlimited polymorphic pointer to a value at the given
        !! coordinates
        procedure, public :: get_value => icon_single_index_get_value
        !> Returns a pointer to a copy of the array of data (this copy is created by
        !! this function using sourced allocation)
        procedure, public :: get_data => icon_single_index_get_data
        !> For each cell in the section
        procedure, public :: for_all_section => icon_single_index_for_all_section
        !> Print the entire  data field
        procedure, public :: print_field_section => icon_single_index_print_field_section
        !> Deallocate the data array
        procedure, public :: deallocate_data => icon_single_index_deallocate_data
        !> Get the number of points
        procedure, public :: get_num_points
end type icon_single_index_field_section

interface icon_single_index_field_section
    procedure :: icon_single_index_field_section_constructor
end interface icon_single_index_field_section

