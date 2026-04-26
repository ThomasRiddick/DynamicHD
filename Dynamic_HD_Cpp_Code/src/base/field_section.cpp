void latlon_set_data_array_element(lat,lon,value) {
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
}

void field_section::set_integer_value(coords* coords_in,int value) {
    class(*), pointer :: pointer_to_value
        allocate(pointer_to_value,source=value)
        call this%set_generic_value(coords_in,pointer_to_value)
        deallocate(pointer_to_value)
}

void set_real_value(coords_in,value) {
    class(field_section) :: this
    class(coords) :: coords_in
    real :: value
    class(*), pointer :: pointer_to_value
        allocate(pointer_to_value,source=value)
        call this%set_generic_value(coords_in,pointer_to_value)
        deallocate(pointer_to_value)
}

void set_logical_value(coords_in,value) {
    class(field_section) :: this
    class(coords) :: coords_in
    logical :: value
    class(*), pointer :: pointer_to_value
        allocate(pointer_to_value,source=value)
        call this%set_generic_value(coords_in,pointer_to_value)
        deallocate(pointer_to_value)
}

function latlon_field_section_constructor(data,section_coords) {
    type(latlon_field_section), pointer :: constructor
    class (*), dimension(:,:), pointer :: data
    class(latlon_section_coords) :: section_coords
        allocate(constructor)
        call constructor%init_latlon_field_section(data,section_coords)
}

void init_latlon_field_section(data,section_coords,wrap_in) {
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
}

pure function latlon_get_value(coords_in) result(value) {
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
}

void latlon_set_generic_value(coords_in,value) {
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
}

void latlon_for_all_section(subroutine_in,calling_object) {
    class(latlon_field_section) :: this
    class(*), intent(inout) :: calling_object
    class(coords), pointer :: coords_in
    integer :: i,j
    interface
        subroutine subroutine_interface(calling_object,coords_in)
            use coords_mod
            class(*), intent(inout) :: calling_object
            class(coords), pointer, intent(inout) :: coords_in
        }
    end interface
    procedure(subroutine_interface) :: subroutine_in
    do j = this%section_min_lon,this%section_max_lon
        do i = this%section_min_lat,this%section_max_lat
            allocate(coords_in,source=latlon_coords(i,j))
            call subroutine_in(calling_object,coords_in)
        end do
    end do
}

pure function get_nlat() result(nlat) {
    class(latlon_field_section), intent(in) :: this
    integer :: nlat
        nlat = this%nlat
}

pure function get_nlon() result(nlon) {
    class(latlon_field_section), intent(in) :: this
    integer :: nlon
        nlon = this%nlon
}

pure function get_wrap() result(wrap) {
    class(latlon_field_section), intent(in) :: this
    logical :: wrap
        wrap = this%wrap
}

//Explicit specification of bounds in source is safer
pure function latlon_get_data() result(data) {
    class(latlon_field_section), intent(in) :: this
    class(*), dimension(:,:), pointer :: data
        allocate(data(size(this%data,1),size(this%data,2)),source=this%data)
}

//Explicit specification of bounds in source is safer
pure function icon_single_index_get_data() result(data) {
    class(icon_single_index_field_section), intent(in) :: this
    class(*), dimension(:), pointer :: data
        allocate(data(size(this%data)),source=this%data)
}

pure function get_section_min_lat() result(section_min_lat) {
    class(latlon_field_section), intent(in) :: this
    integer section_min_lat
        section_min_lat = this%section_min_lat
}

pure function get_section_min_lon() result(section_min_lon) {
    class(latlon_field_section), intent(in) :: this
    integer section_min_lon
        section_min_lon = this%section_min_lon
}

pure function get_section_max_lat() result(section_max_lat) {
    class(latlon_field_section), intent(in) :: this
    integer section_max_lat
        section_max_lat = this%section_max_lat
}

pure function get_section_max_lon() result(section_max_lon) {
    class(latlon_field_section), intent(in) :: this
    integer section_max_lon
        section_max_lon = this%section_max_lon
}

void latlon_print_field_section() {
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
}

void latlon_deallocate_data() {
    class(latlon_field_section) :: this
        deallocate(this%data)
}

icon_single_index_field_section::icon_single_index_field_section(data,section_coords) {
    type(icon_single_index_field_section), pointer :: constructor
    class (*), dimension(:), pointer :: data
    class(generic_1d_section_coords) section_coords
        allocate(constructor)
        call constructor%init_icon_single_index_field_section(data,section_coords)
}

icon_single_index_field_section::init_icon_single_index_field_section(input_data,section_coords) {
    class(*), dimension(:), pointer :: input_data
    type(generic_1d_section_coords) :: section_coords
        this%data => input_data
        this%cell_neighbors => section_coords%get_cell_neighbors()
        this%cell_secondary_neighbors => &
            section_coords%get_cell_secondary_neighbors()
        this%num_points = size(input_data)
        this%mask => section_coords%get_section_mask()
}

void icon_single_index_set_data_array_element(index,value) {
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
}

pure function icon_single_index_get_value(coords_in) result(value) {
    class(icon_single_index_field_section), intent(in) :: this
    class(coords), intent(in) :: coords_in
    class(*), pointer :: value
        select type (coords_in)
        type is (generic_1d_coords)
            allocate(value,source=this%data(coords_in%index))
        end select
}

void icon_single_index_set_generic_value(coords_in,value) {
    class(icon_single_index_field_section) :: this
    class(coords), intent(in) :: coords_in
    class(*), pointer, intent(in) :: value
        select type (coords_in)
        type is (generic_1d_coords)
            call this%set_data_array_element(coords_in%index,value)
        end select
}

void icon_single_index_for_all_section(subroutine_in,calling_object) {
    class(icon_single_index_field_section) :: this
    class(*), intent(inout) :: calling_object
    class(coords), pointer :: coords_in
    integer :: i
    interface
        subroutine subroutine_interface(calling_object,coords_in)
            use coords_mod
            class(*), intent(inout) :: calling_object
            class(coords), pointer, intent(inout) :: coords_in
        }
    end interface
    procedure(subroutine_interface) :: subroutine_in
        do i = 1,this%num_points
            if (this%mask(i)) then
                allocate(coords_in,source=generic_1d_coords(i,.true.))
                call subroutine_in(calling_object,coords_in)
            end if
        end do
}

void icon_single_index_print_field_section() {
    class(icon_single_index_field_section) :: this
    integer :: i
        select type(data => this%data)
        type is (integer)
                write (*,*) (data(i),i=1,size(data))
        type is (logical)
                write (*,*) (data(i),i=1,size(data))
        end select
}

function get_mask() result(mask) {
    class(icon_single_index_field_section) :: this
    logical, dimension(:), pointer :: mask
    mask => this%mask
}

int icon_single_index_field_section::get_num_points() {
    return num_points
}

void icon_single_index_deallocate_data() {
    class(icon_single_index_field_section) :: this
        deallocate(this%data)
}
