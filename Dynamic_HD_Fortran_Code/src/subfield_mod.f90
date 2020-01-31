module subfield_mod
use coords_mod
implicit none
private
public :: latlon_subfield_constructor,icon_single_index_subfield_constructor

!> A class to contain a subfield. Unlike a field section a subfield only has a data array
!! that covers grid points within the selected section of field. Coordinates should be
!! specified on the whole grid however and are converted position within subfields data
!! array by adding specified offsets.
type, public, abstract :: subfield
contains
    !> Class destructor; deallocates the array holding the classes data.
    procedure(destructor), deferred :: destructor
    !> Wrapper to a set an integer value at given set of coordinates
    procedure :: set_integer_value
    !> Wrapper to a set a real value at given set of coordinates
    procedure :: set_real_value
    !> Wrapper to a set a logical value at given set of coordinates
    procedure :: set_logical_value
    !> Wrapper to a set a coords value at given set of coordinates
    procedure :: set_coords_value
    !> Generic type bound procedure to set a given value at a given coordinate
    !! this then calls one of the type specific wrappers for setting values
    generic :: set_value => set_integer_value, set_real_value, set_logical_value, &
                            set_coords_value
    !> Wrapper to a set an integer value for the entire field
    procedure :: set_all_integer
    !> Wrapper to a set a real value for the entire field
    procedure :: set_all_real
    !> Wrapper to a set a logical value for the entire field
    procedure :: set_all_logical
    !> Generic type bound procedure to set a given value for the entire field
    !! this then calls one of the type specific wrappers for setting values
    generic :: set_all => set_all_integer, set_all_real, set_all_logical
    !> Given a pointer to an unlimited polymorphic variable set it at the given
    !! coordinates
    procedure(set_generic_value), deferred :: set_generic_value
    !> Get an unlimited polymorphic pointer to a value at the given coordinates
    procedure(get_value), deferred :: get_value
    !> Process all cells of field
    procedure(for_all), deferred :: for_all
    !> Iterate over all the cells along the edges of this subfield
    procedure(for_all_edge_cells), deferred :: for_all_edge_cells
    !> Given a pointer to an unlimited polymorphic variable set the entire field
    !! to that value
    procedure(set_all_generic), deferred :: set_all_generic
    !> Check if given coordinates are outside this subfield
    procedure(coords_outside_subfield), deferred :: coords_outside_subfield
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

    subroutine for_all(this,subroutine_in,calling_object)
        import subfield
        implicit none
        class(subfield) :: this
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords),intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        class(*), intent(inout) :: calling_object
    end subroutine for_all

    subroutine for_all_edge_cells(this,subroutine_in,calling_object)
        import subfield
        implicit none
        class(subfield) :: this
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords), intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        class(*), intent(inout) :: calling_object
    end subroutine for_all_edge_cells

    subroutine set_all_generic(this,value)
        import subfield
        implicit none
        class(subfield) :: this
        class(*), pointer :: value
    end subroutine set_all_generic

    function coords_outside_subfield(this,coords_in) result (is_true)
        import subfield
        import coords
        implicit none
        class(subfield) :: this
        class(coords), pointer :: coords_in
        logical :: is_true
    end function coords_outside_subfield
end interface

!> A concrete subclass of subfield for a latitude longitude grid
type, extends(subfield), public :: latlon_subfield
    private
    !> A 2D array to hold the latitude longitude subfield data
    class (*), dimension(:,:), pointer :: data => null()
    !> Latitude offset to remove from input coordinates on the full grid to
    !! get coordinates within the grid of the subfield
    integer :: lat_offset
    !> Longitude offset to remove from input coordinates on the full grid to
    !! get coordinates within the grid of the subfield
    integer :: lon_offset
    !> Minimum latitude in this subfield
    integer :: lat_min
    !> Minimum longitude in this subfield
    integer :: lon_min
    !> Maximum latitude in this subfield
    integer :: lat_max
    !> Maximum longitude in this subfield
    integer :: lon_max
    !> Wrap coordinates in this subfield... this should be false unless the subfield
    !> span 360 degree of longitude
    logical :: wrap
contains
    !> Initialize a latitude longitude subfield; takes a pointer to an array of
    !! data that fits the subfield and a section coordinates object that specifies
    !! the position of the subfield within the main field
    procedure, private :: init_latlon_subfield
    !> Get value in the subfield from a given set of lat-lon coordinates on the
    !! main grid (which are then coverted to subfield coordinates by the using
    !! the lat and lon offsets)
    procedure :: get_value => latlon_get_value
    !> Given a pointer to an unlimited polymorphic variable set it at the given
    !! latitude logitude coordinates on the main grid (converting to the
    !! coordinates of the subfield using the latitude and longitude offsets)
    procedure :: set_generic_value => latlon_set_generic_value
    !In leiu of a final routine while this is not support for
    !all fortran compilers
    !final :: destructor
    !> Class destructor; deallocates the data array if it exists
    procedure :: destructor => latlon_subfield_destructor
    !> Set an element of the data array using the internal latitude and
    !! longitude coordinates (as individual numbers) of the subfields grid.
    !! The third argument is a value; this must be of the same type as this
    !! classes data member variable
    procedure, private :: set_data_array_element => latlon_set_data_array_element
    !> Iterate over all the cells in this subfield
    procedure :: for_all => latlon_for_all
    !> Iterate over all the cells along the edges of this subfield
    procedure :: for_all_edge_cells => latlon_for_all_edge_cells
    !> Returns a pointer to a copy of the array of data (this copy is created by
    !! this function using sourced allocation)
    procedure :: latlon_get_data
    !> Given a pointer to an unlimited polymorphic variable set the entire field
    !! to that value
    procedure :: set_all_generic => latlon_set_all_generic
    !> Check if given coordinates are outside this subfield
    procedure :: coords_outside_subfield => latlon_coords_outside_subfield
end type latlon_subfield

interface latlon_subfield
    procedure latlon_subfield_constructor
end interface latlon_subfield

!> A concrete subclass of subfield for an icon single index grid
type, extends(subfield), public :: icon_single_index_subfield
    private
    !> A 1D array to hold the subfield data
    class (*), dimension(:), pointer :: data => null()
    integer, dimension(:,:), pointer :: cell_neighbors
    integer, dimension(:,:), pointer :: cell_secondary_neighbors
    integer, dimension(:), pointer :: edge_cells
    integer, dimension(:), pointer :: subfield_indices
    logical, dimension(:), pointer :: mask
    type(generic_1d_coords) :: outside_subfield_coords
    integer :: num_points
    integer :: num_edge_cells
contains
    procedure, private :: init_icon_single_index_subfield
    procedure :: get_value => icon_single_index_get_value
    procedure :: set_generic_value => icon_single_index_set_generic_value
    procedure :: destructor => icon_single_index_subfield_destructor
    procedure, private :: set_data_array_element => icon_single_index_set_data_array_element
    procedure :: for_all => icon_single_index_for_all
    procedure :: for_all_edge_cells => icon_single_index_for_all_edge_cells
    procedure :: icon_single_index_get_data
    procedure :: set_all_generic => icon_single_index_set_all_generic
    procedure :: coords_outside_subfield => icon_single_index_coords_outside_subfield
end type icon_single_index_subfield

interface icon_single_index_subfield
    procedure icon_single_index_subfield_constructor
end interface icon_single_index_subfield

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

    subroutine set_coords_value(this,coords_in,value)
        class(subfield), intent(in) :: this
        class(coords) :: coords_in
        class(coords), intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_generic_value(coords_in,pointer_to_value)
    end subroutine

    subroutine set_all_integer(this,value)
        class(subfield),intent(in) :: this
        integer, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_all_generic(pointer_to_value)
    end subroutine

    subroutine set_all_real(this,value)
        class(subfield),intent(in) :: this
        real, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_all_generic(pointer_to_value)
    end subroutine

    subroutine set_all_logical(this,value)
        class(subfield), intent(in) :: this
        logical, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%set_all_generic(pointer_to_value)
    end subroutine

    function latlon_subfield_constructor(input_data,subfield_section_coords,wrap) &
            result(constructor)
        type(latlon_subfield), pointer :: constructor
        class(*), dimension(:,:), pointer :: input_data
        class(latlon_section_coords) :: subfield_section_coords
        logical, optional :: wrap
            allocate(constructor)
            if (present(wrap)) then
                call constructor%init_latlon_subfield(input_data,subfield_section_coords,wrap)
            else
                call constructor%init_latlon_subfield(input_data,subfield_section_coords)
            end if
    end function latlon_subfield_constructor

    subroutine init_latlon_subfield(this,input_data,subfield_section_coords,wrap)
        class(latlon_subfield) :: this
        class(*), dimension(:,:), pointer :: input_data
        class(latlon_section_coords) :: subfield_section_coords
        logical, optional :: wrap
            if (present(wrap)) then
                this%wrap = wrap
            else
                this%wrap = .false.
            end if
            this%data => input_data
            this%lat_offset = subfield_section_coords%section_min_lat - 1
            this%lon_offset = subfield_section_coords%section_min_lon - 1
            this%lat_min    = subfield_section_coords%section_min_lat
            this%lat_max    = subfield_section_coords%section_width_lat + &
                              this%lat_offset
            this%lon_min    = subfield_section_coords%section_min_lon
            this%lon_max    = subfield_section_coords%section_width_lon + &
                              this%lon_offset
    end subroutine init_latlon_subfield

    pure function latlon_get_value(this,coords_in) result(value)
        class(latlon_subfield), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
        integer :: lat, lon
            select type (coords_in)
            type is (latlon_coords)
                if ( this%lat_max >= coords_in%lat .and. coords_in%lat >= this%lat_min) then
                    lat = coords_in%lat
                else if (this%lat_max < coords_in%lat) then
                    lat = this%lat_max
                else
                    lat = this%lat_min
                end if
                if ( this%lon_max >= coords_in%lon .and. coords_in%lon >= this%lon_min) then
                    lon = coords_in%lon
                else if (this%lon_max < coords_in%lon) then
                    if (.not. this%wrap) then
                        lon = this%lon_max
                    else
                        lon = coords_in%lon - this%lon_max
                    end if
                else
                    if (.not. this%wrap) then
                        lon = this%lon_min
                    else
                        lon = this%lon_max - this%lon_min + 1 + coords_in%lon
                    end if
                end if
                allocate(value,source=this%data(lat-this%lat_offset,&
                                                lon-this%lon_offset))
            end select
    end function latlon_get_value

    subroutine latlon_set_generic_value(this,coords_in,value)
        class(latlon_subfield) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
        integer :: lat,lon
            select type (coords_in)
            type is (latlon_coords)
                if ( this%lat_max >= coords_in%lat .and. coords_in%lat >= this%lat_min) then
                    lat = coords_in%lat
                else if (this%lat_max < coords_in%lat) then
                    lat = this%lat_max
                else
                    lat = this%lat_min
                end if
                if ( this%lon_max >= coords_in%lon .and. coords_in%lon >= this%lon_min) then
                    lon = coords_in%lon
                else if (this%lon_max < coords_in%lon) then
                    if (.not. this%wrap) then
                        lon = this%lon_max
                    else
                        lon = coords_in%lon - this%lon_max
                    end if
                else
                    if (.not. this%wrap) then
                        lon = this%lon_min
                    else
                        lon = this%lon_max - this%lon_min + 1 + coords_in%lon
                    end if
                end if
                call this%set_data_array_element(lat-this%lat_offset,&
                    lon-this%lon_offset,value)
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

    subroutine latlon_for_all(this,subroutine_in,calling_object)
        implicit none
        class(latlon_subfield) :: this
        class(*), intent(inout) :: calling_object
        integer :: i,j
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords), intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        do j = this%lon_min,this%lon_max
            do i = this%lat_min,this%lat_max
                call subroutine_in(calling_object,latlon_coords(i,j))
            end do
        end do
    end subroutine latlon_for_all

    subroutine latlon_for_all_edge_cells(this,subroutine_in,calling_object)
        implicit none
        class(latlon_subfield) :: this
        class(*), intent(inout) :: calling_object
        integer :: i,j
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords),intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
        do j = this%lon_min,this%lon_max
                call subroutine_in(calling_object,latlon_coords(this%lat_min,j))
                call subroutine_in(calling_object,latlon_coords(this%lat_max,j))
        end do
        do i = this%lat_min+1,this%lat_max-1
                call subroutine_in(calling_object,latlon_coords(i,this%lon_min))
                call subroutine_in(calling_object,latlon_coords(i,this%lon_max))
        end do
    end subroutine latlon_for_all_edge_cells

    subroutine latlon_set_all_generic(this,value)
        class(latlon_subfield) :: this
        class(*), pointer :: value
        integer :: i,j
        do j = this%lon_min,this%lon_max
            do i = this%lat_min,this%lat_max
                call this%set_generic_value(latlon_coords(i,j),value)
            end do
        end do
    end subroutine latlon_set_all_generic

    function latlon_coords_outside_subfield(this,coords_in) result(is_true)
        implicit none
        class(latlon_subfield) :: this
        class(coords), pointer :: coords_in
        logical :: is_true
            select type (coords_in)
            type is (latlon_coords)
                is_true = ( coords_in%lat < this%lat_min .or. &
                            coords_in%lat > this%lat_max .or. &
                            coords_in%lon < this%lon_min .or. &
                            coords_in%lon > this%lon_max )
            end select
    end function latlon_coords_outside_subfield

    subroutine icon_single_index_set_data_array_element(this,index,value)
        class(icon_single_index_subfield) :: this
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

    function icon_single_index_subfield_constructor(input_data,subfield_section_coords) &
            result(constructor)
        type(icon_single_index_subfield), pointer :: constructor
        class(*), dimension(:), pointer :: input_data
        class(generic_1d_section_coords) :: subfield_section_coords
            allocate(constructor)
            call constructor%init_icon_single_index_subfield(input_data,subfield_section_coords)
    end function icon_single_index_subfield_constructor

    subroutine init_icon_single_index_subfield(this,input_data,subfield_section_coords)
        class(icon_single_index_subfield) :: this
        class(*), dimension(:), pointer :: input_data
        type(generic_1d_section_coords) :: subfield_section_coords
            this%data => input_data
            this%cell_neighbors => subfield_section_coords%get_cell_neighbors()
            this%cell_secondary_neighbors => &
                subfield_section_coords%get_cell_secondary_neighbors()
    end subroutine init_icon_single_index_subfield

    pure function icon_single_index_get_value(this,coords_in) result(value)
        class(icon_single_index_subfield), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
        integer :: converted_index
            select type (coords_in)
            type is (generic_1d_coords)
                if(coords_in%use_internal_index) then
                    allocate(value,source=this%data(coords_in%index))
                else
                    converted_index = this%subfield_indices(coords_in%index)
                    allocate(value,source=this%data(converted_index))
                end if
            end select
    end function icon_single_index_get_value

    subroutine icon_single_index_set_generic_value(this,coords_in,value)
        class(icon_single_index_subfield) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: value
        integer :: converted_index
            select type (coords_in)
            type is (generic_1d_coords)
                if(coords_in%use_internal_index) then
                    call this%set_data_array_element(coords_in%index,value)
                else
                    converted_index = this%subfield_indices(coords_in%index)
                    call this%set_data_array_element(converted_index,value)
                end if
            end select
            deallocate(value)
    end subroutine icon_single_index_set_generic_value

    subroutine icon_single_index_subfield_destructor(this)
        class(icon_single_index_subfield) :: this
            if (associated(this%data)) deallocate(this%data)
    end subroutine

    pure function icon_single_index_get_data(this) result(data)
        class(icon_single_index_subfield), intent(in) :: this
        class(*), dimension(:), pointer :: data
            allocate(data(size(this%data)),source=this%data)
    end function icon_single_index_get_data

    subroutine icon_single_index_for_all(this,subroutine_in,calling_object)
        implicit none
        class(icon_single_index_subfield) :: this
        class(*), intent(inout) :: calling_object
        integer :: i
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords), intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
            do i = 1,this%num_points
                call subroutine_in(calling_object,generic_1d_coords(i,.true.))
            end do
    end subroutine icon_single_index_for_all

    subroutine icon_single_index_for_all_edge_cells(this,subroutine_in,calling_object)
        implicit none
        class(icon_single_index_subfield) :: this
        class(*), intent(inout) :: calling_object
        integer :: i
        interface
            subroutine subroutine_interface(calling_object,coords_in)
                use coords_mod
                class(*), intent(inout) :: calling_object
                class(coords),intent(in) :: coords_in
            end subroutine subroutine_interface
        end interface
        procedure(subroutine_interface) :: subroutine_in
            do i = 1,this%num_edge_cells
                    call subroutine_in(calling_object,generic_1d_coords(this%edge_cells(i),.true.))
            end do
    end subroutine icon_single_index_for_all_edge_cells

    subroutine icon_single_index_set_all_generic(this,value)
        class(icon_single_index_subfield) :: this
        class(*), pointer :: value
        integer :: i
            do i = 1,this%num_points
                call this%set_generic_value(generic_1d_coords(i,.true.),value)
            end do
    end subroutine icon_single_index_set_all_generic


    function icon_single_index_coords_outside_subfield(this,coords_in) result(is_true)
        implicit none
        class(icon_single_index_subfield) :: this
        class(coords), pointer :: coords_in
        logical :: is_true
        integer :: converted_index
            select type (coords_in)
            type is (generic_1d_coords)
                if(coords_in%use_internal_index) then
                    is_true = (coords_in%are_equal_to(this%outside_subfield_coords))
                else
                    converted_index = this%subfield_indices(coords_in%index)
                    is_true = (this%outside_subfield_coords%index == converted_index)
                end if
            end select
    end function icon_single_index_coords_outside_subfield

end module subfield_mod
