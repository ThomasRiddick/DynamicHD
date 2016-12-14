module subfield_mod
use coords_mod
implicit none
private
public :: latlon_subfield_constructor

type, public, abstract :: subfield
contains
    procedure(destructor), deferred :: destructor
    procedure :: set_integer_value
    procedure :: set_real_value
    procedure :: set_logical_value
    generic :: set_value => set_integer_value, set_real_value, set_logical_value
    procedure(set_generic_value), deferred :: set_generic_value
    procedure(get_value), deferred :: get_value
end type subfield

abstract interface
    pure function get_value(this,coords_in) result(value)
        import subfield
        import coords
        implicit none
        class(subfield), intent(in) :: this
        class(coords), intent(in)   :: coords_in
        class(*), pointer :: value
    end function

    subroutine set_generic_value(this,coords_in,value)
        import subfield
        import coords
        implicit none
        class(subfield) :: this
        class(coords),intent(in) :: coords_in
        class(*), pointer :: value
    end subroutine

    subroutine destructor(this)
        import subfield
        implicit none
        class(subfield) :: this
    end subroutine
end interface

type, extends(subfield), public :: latlon_subfield
    private
    class (*), dimension(:,:), pointer :: data => null()
    integer :: lat_offset
    integer :: lon_offset
contains
    procedure, private :: init_latlon_subfield
    procedure :: get_value => latlon_get_value
    procedure :: set_generic_value => latlon_set_generic_value
    !In leiu of a final routine while this is not support for
    !all fortran compilers
    !final :: destructor
    procedure :: destructor => latlon_subfield_destructor
    procedure, private :: set_data_array_element => latlon_set_data_array_element
    procedure :: latlon_get_data
end type latlon_subfield

interface latlon_subfield
    procedure latlon_subfield_constructor
end interface latlon_subfield

contains

    subroutine latlon_set_data_array_element(this,lat,lon,value)
        class(latlon_subfield) :: this
        class(*), pointer :: value
        integer :: lat,lon
        select type (value)
        type is (integer)
            select type (data => this%data)
                type is (integer)
                    data(lat,lon) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select
        type is (logical)
            select type (data => this%data)
                type is (logical)
                    data(lat,lon) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select
        type is (real)
            select type (data => this%data)
                type is (real)
                    data(lat,lon) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select
        class default
            stop 'trying to set array element with value of a unknown type'
        end select
    end subroutine latlon_set_data_array_element

    subroutine set_integer_value(this,coords_in,value)
        class(subfield),intent(in) :: this
        class(coords) :: coords_in
        integer, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
    end subroutine
    
    subroutine set_real_value(this,coords_in,value)
        class(subfield),intent(in) :: this
        class(coords) :: coords_in
        real, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
    end subroutine

    subroutine set_logical_value(this,coords_in,value)
        class(subfield), intent(in) :: this
        class(coords) :: coords_in
        logical, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
    end subroutine

    function latlon_subfield_constructor(input_data,subfield_section_coords) &
            result(constructor)
        type(latlon_subfield), pointer :: constructor
        class(*), dimension(:,:), pointer :: input_data
        class(latlon_section_coords) :: subfield_section_coords
            allocate(constructor)
            call constructor%init_latlon_subfield(input_data,subfield_section_coords)
    end function latlon_subfield_constructor

    subroutine init_latlon_subfield(this,input_data,subfield_section_coords)
        class(latlon_subfield) :: this
        class(*), dimension(:,:), pointer :: input_data
        type(latlon_section_coords) :: subfield_section_coords
            this%data => input_data
            this%lat_offset = subfield_section_coords%section_min_lat - 1
            this%lon_offset = subfield_section_coords%section_min_lon - 1
    end subroutine init_latlon_subfield

    pure function latlon_get_value(this,coords_in) result(value)
        class(latlon_subfield), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
            select type (coords_in)
            type is (latlon_coords)
                allocate(value,source=this%data(coords_in%lat-this%lat_offset,&
                                                coords_in%lon-this%lon_offset))
            end select
    end function latlon_get_value

    subroutine latlon_set_generic_value(this,coords_in,value)
        class(latlon_subfield) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
            select type (coords_in)
            type is (latlon_coords)
                call this%set_data_array_element(coords_in%lat-this%lat_offset,&
                    coords_in%lon-this%lon_offset,value)
            end select
            deallocate(value)
    end subroutine latlon_set_generic_value

    subroutine latlon_subfield_destructor(this)
        class(latlon_subfield) :: this
            if (associated(this%data)) deallocate(this%data)
    end subroutine

    pure function latlon_get_data(this) result(data)
        class(latlon_subfield), intent(in) :: this
        class(*), dimension(:,:), pointer :: data
            allocate(data(size(this%data,1),size(this%data,2)),source=this%data)
    end function latlon_get_data

end module subfield_mod
