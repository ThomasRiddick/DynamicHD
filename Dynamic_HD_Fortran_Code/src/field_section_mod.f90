module field_section_mod
use coords_mod
implicit none
private
public :: latlon_field_section_constructor

type, public, abstract :: field_section
    contains
    procedure :: set_integer_value
    procedure :: set_real_value
    procedure :: set_logical_value
    generic :: set_value => set_integer_value, set_real_value, set_logical_value
    procedure(get_value), deferred :: get_value
    procedure(set_generic_value), deferred :: set_generic_value
    procedure(print_field_section), deferred :: print_field_section
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

type, extends(field_section), public :: latlon_field_section
    class(*), dimension (:,:), pointer :: data
    integer                  :: nlat
    integer                  :: nlon
    integer                  :: section_min_lat
    integer                  :: section_min_lon
    integer                  :: section_max_lat
    integer                  :: section_max_lon
    contains
        private
        procedure :: init_latlon_field_section
        procedure :: set_data_array_element => latlon_set_data_array_element
        procedure, public :: get_nlat
        procedure, public :: get_nlon
        procedure, public :: get_data
        procedure, public :: get_section_min_lat
        procedure, public :: get_section_min_lon
        procedure, public :: get_section_max_lat
        procedure, public :: get_section_max_lon
        procedure, public :: set_generic_value => latlon_set_generic_value
        procedure, public :: get_value => latlon_get_value
        procedure, public :: print_field_section => latlon_print_field_section
        procedure, public :: deallocate_data => latlon_deallocate_data
end type latlon_field_section

interface latlon_field_section
    procedure latlon_field_section_constructor
end interface latlon_field_section

contains

    subroutine latlon_set_data_array_element(this,lat,lon,value)
        class(latlon_field_section) :: this
        class(*), pointer :: value
        integer :: lat,lon
        select type (value)
        type is (integer)
            select type (data=>this%data)
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
        class(field_section) :: this
        class(coords) :: coords_in
        integer :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
            deallocate(pointer_to_value)
    end subroutine set_integer_value

   subroutine set_real_value(this,coords_in,value)
        class(field_section) :: this
        class(coords) :: coords_in
        real :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
            deallocate(pointer_to_value)
    end subroutine set_real_value

   subroutine set_logical_value(this,coords_in,value)
        class(field_section) :: this
        class(coords) :: coords_in
        logical :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
            deallocate(pointer_to_value)
    end subroutine set_logical_value

    function latlon_field_section_constructor(data,section_coords) result(constructor)
        type(latlon_field_section), pointer :: constructor
        class (*), dimension(:,:), pointer :: data
        class(latlon_section_coords) section_coords
            allocate(constructor)
            call constructor%init_latlon_field_section(data,section_coords)
    end function latlon_field_section_constructor

    subroutine init_latlon_field_section(this,data,section_coords)
        class(latlon_field_section) :: this
        class(*), dimension(:,:), pointer :: data
        class(latlon_section_coords) section_coords
            this%data => data
            this%nlat = SIZE(data,1)
            this%nlon = SIZE(data,2)
            this%section_min_lat = section_coords%section_min_lat
            this%section_min_lon = section_coords%section_min_lon
            this%section_max_lat = section_coords%section_min_lat + section_coords%section_width_lat - 1
            this%section_max_lon = section_coords%section_min_lon + section_coords%section_width_lon - 1
    end subroutine init_latlon_field_section

    pure function latlon_get_value(this,coords_in) result(value)
        class(latlon_field_section), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
        integer :: lat,lon
            select type(coords_in)
                type is (latlon_coords)
                if ( this%nlat >= coords_in%lat .and. coords_in%lat > 0) then
                    lat = coords_in%lat
                else if (this%nlat < coords_in%lat) then
                    lat = this%nlat
                else
                    lat = 0
                end if
                if ( this%nlon >= coords_in%lon .and. coords_in%lon > 0) then
                    lon = coords_in%lon
                else if (this%nlon < coords_in%lon) then
                    lon = coords_in%lon - this%nlon
                else
                    lon = this%nlon + coords_in%lon
                end if
            end select
            allocate(value,source=this%data(lat,lon))
    end function latlon_get_value

    subroutine latlon_set_generic_value(this,coords_in,value)
        class(latlon_field_section) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer, intent(in) :: value
            select type (coords_in)
            type is (latlon_coords)
                if ( this%nlon >= coords_in%lon .and. coords_in%lon > 0) then
                    call this%set_data_array_element(coords_in%lat,coords_in%lon,value)
                else if (this%nlon < coords_in%lon) then
                    call this%set_data_array_element(coords_in%lat,coords_in%lon - this%nlon,value)
                else
                    call this%set_data_array_element(coords_in%lat,this%nlon + coords_in%lon,value)
                end if
            end select
    end subroutine latlon_set_generic_value

    pure function get_nlat(this) result(nlat)
        class(latlon_field_section), intent(in) :: this
        integer nlat
            nlat = this%nlat
    end function get_nlat

    pure function get_nlon(this) result(nlon)
        class(latlon_field_section), intent(in) :: this
        integer nlon
            nlon = this%nlon
    end function get_nlon

    pure function get_data(this) result(data)
        class(latlon_field_section), intent(in) :: this
        class(*), dimension(:,:), pointer :: data
            allocate(data(size(this%data,1),size(this%data,2)),source=this%data)
    end function get_data

    pure function get_section_min_lat(this) result(section_min_lat)
        class(latlon_field_section), intent(in) :: this
        integer section_min_lat
            section_min_lat = this%section_min_lat
    end function get_section_min_lat

    pure function get_section_min_lon(this) result(section_min_lon)
        class(latlon_field_section), intent(in) :: this
        integer section_min_lon
            section_min_lon = this%section_min_lon
    end function get_section_min_lon

    pure function get_section_max_lat(this) result(section_max_lat)
        class(latlon_field_section), intent(in) :: this
        integer section_max_lat
            section_max_lat = this%section_max_lat
    end function get_section_max_lat

    pure function get_section_max_lon(this) result(section_max_lon)
        class(latlon_field_section), intent(in) :: this
        integer section_max_lon
            section_max_lon = this%section_max_lon
    end function get_section_max_lon

    subroutine latlon_print_field_section(this)
        class(latlon_field_section) :: this
        integer :: i,j
            select type(data => this%data)
            type is (integer)
                do i = 1,size(data,1)
                    write (*,*) (data(i,j),j=1,size(data,2))
                end do
            type is (logical)
                do i = 1,size(data,1)
                    write (*,*) (data(i,j),j=1,size(data,2))
                end do
            end select
    end subroutine latlon_print_field_section

    subroutine latlon_deallocate_data(this)
        class(latlon_field_section) :: this
            deallocate(this%data)
    end subroutine latlon_deallocate_data

end module field_section_mod
