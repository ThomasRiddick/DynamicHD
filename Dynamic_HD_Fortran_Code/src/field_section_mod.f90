module field_section_mod
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
                class(*) :: calling_object
                class(coords),intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        class(*) :: calling_object
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
        !> Returns a pointer to a copy of the array of data (this copy is created by
        !! this function using sourced allocation)
        procedure, public :: get_data
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
        class(latlon_section_coords) :: section_coords
            allocate(constructor)
            call constructor%init_latlon_field_section(data,section_coords)
    end function latlon_field_section_constructor

    subroutine init_latlon_field_section(this,data,section_coords,wrap_in)
        class(latlon_field_section) :: this
        class(*), dimension(:,:), pointer :: data
        class(latlon_section_coords) section_coords
        logical, optional :: wrap_in
        logical           :: wrap
            if(present(wrap_in)) then
                wrap = wrap_in
            else
                wrap = .true.
            end if
            this%data => data
            this%nlat = SIZE(data,1)
            this%nlon = SIZE(data,2)
            this%section_min_lat = section_coords%section_min_lat
            this%section_min_lon = section_coords%section_min_lon
            this%section_max_lat = section_coords%section_min_lat + section_coords%section_width_lat - 1
            this%section_max_lon = section_coords%section_min_lon + section_coords%section_width_lon - 1
            this%wrap = wrap
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
                    lat = 1
                end if
                if ( this%nlon >= coords_in%lon .and. coords_in%lon > 0) then
                    lon = coords_in%lon
                else if (this%nlon < coords_in%lon) then
                    if (.not. this%wrap) then
                        lon = this%nlon
                    else
                        lon = coords_in%lon - this%nlon
                    end if
                else
                    if (.not. this%wrap) then
                        lon = 1
                    else
                        lon = this%nlon + coords_in%lon
                    end if
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
                    if (.not. this%wrap) then
                        stop 'Trying to write element outside of field boundries'
                    else
                        call this%set_data_array_element(coords_in%lat,coords_in%lon - this%nlon,value)
                    end if
                else
                    if (.not. this%wrap) then
                        stop 'Trying to write element outside of field boundries'
                    else
                        call this%set_data_array_element(coords_in%lat,this%nlon + coords_in%lon,value)
                    end if
                end if
            end select
    end subroutine latlon_set_generic_value

    subroutine latlon_for_all_section(this,subroutine_in,calling_object)
        implicit none
        class(latlon_field_section) :: this
        class(*) :: calling_object
        integer :: i,j
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*) :: calling_object
                class(coords), intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        do j = this%section_min_lon,this%section_max_lon
            do i = this%section_min_lat,this%section_max_lat
                call subroutine_in(calling_object,latlon_coords(i,j))
            end do
        end do
    end subroutine latlon_for_all_section

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

    function icon_single_index_field_section_constructor(data,section_coords) result(constructor)
        type(icon_single_index_field_section), pointer :: constructor
        class (*), dimension(:), pointer :: data
        class(generic_1d_section_coords) section_coords
            allocate(constructor)
            call constructor%init_icon_single_index_field_section(data,section_coords)
    end function icon_single_index_field_section_constructor

    subroutine init_icon_single_index_field_section(this,input_data,section_coords)
        class(icon_single_index_field_section) :: this
        class(*), dimension(:), pointer :: input_data
        type(generic_1d_section_coords) :: section_coords
            this%data => input_data
            this%cell_neighbors => section_coords%get_cell_neighbors()
            this%cell_secondary_neighbors => &
                section_coords%get_cell_secondary_neighbors()
            this%num_points = size(input_data)
            this%mask => section_coords%get_section_mask()
    end subroutine init_icon_single_index_field_section

    subroutine icon_single_index_set_data_array_element(this,index,value)
        class(icon_single_index_field_section) :: this
        class(*), pointer :: value
        integer :: index
        select type (value)
        type is (integer)
            select type (data => this%data)
                type is (integer)
                    data(index) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select
        type is (logical)
            select type (data => this%data)
                type is (logical)
                    data(index) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select
        type is (real)
            select type (data => this%data)
                type is (real)
                    data(index) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select
        type is (generic_1d_coords)
            select type (data=> this%data)
                type is (generic_1d_coords)
                    data(index) = value
                class default
                    stop 'trying to set array element with value of incorrect type'
            end select

        class default
            stop 'trying to set array element with value of a unknown type'
        end select
    end subroutine icon_single_index_set_data_array_element

    pure function icon_single_index_get_value(this,coords_in) result(value)
        class(icon_single_index_field_section), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
            select type (coords_in)
            type is (generic_1d_coords)
                allocate(value,source=this%data(coords_in%index))
            end select
    end function icon_single_index_get_value

    subroutine icon_single_index_set_generic_value(this,coords_in,value)
        class(icon_single_index_field_section) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer, intent(in) :: value
            select type (coords_in)
            type is (generic_1d_coords)
                call this%set_data_array_element(coords_in%index,value)
            end select
    end subroutine icon_single_index_set_generic_value

    subroutine icon_single_index_for_all_section(this,subroutine_in,calling_object)
        implicit none
        class(icon_single_index_field_section) :: this
        class(*) :: calling_object
        integer :: i
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*) :: calling_object
                class(coords), intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
            do i = 1,this%num_points
                if (this%mask(i)) then
                    call subroutine_in(calling_object,generic_1d_coords(i,.true.))
                end if
            end do
    end subroutine icon_single_index_for_all_section

    subroutine icon_single_index_print_field_section(this)
        class(icon_single_index_field_section) :: this
        integer :: i
            select type(data => this%data)
            type is (integer)
                    write (*,*) (data(i),i=1,size(data))
            type is (logical)
                    write (*,*) (data(i),i=1,size(data))
            end select
    end subroutine icon_single_index_print_field_section

    function get_mask(this) result(mask)
        class(icon_single_index_field_section) :: this
        logical, dimension(:), pointer :: mask
        mask => this%mask
    end function get_mask

    function get_num_points(this) result(npoints)
        class(icon_single_index_field_section) :: this
        integer :: npoints
            npoints = this%num_points
    end function

    subroutine icon_single_index_deallocate_data(this)
        class(icon_single_index_field_section) :: this
            deallocate(this%data)
    end subroutine icon_single_index_deallocate_data

end module field_section_mod
