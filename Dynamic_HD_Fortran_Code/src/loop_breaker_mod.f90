module loop_breaker_mod
    use coords_mod
    use field_section_mod
    use doubly_linked_list_mod
    implicit none
    private
    public :: latlon_dir_based_rdirs_loop_breaker_constructor

    type, abstract, public :: loop_breaker
        private
        class(field_section), pointer :: course_catchment_field => null()
        class(field_section), pointer :: course_cumulative_flow_field => null()
        class(field_section), pointer :: course_rdirs_field => null()
        class(field_section), pointer :: fine_rdirs_field => null()
        class(field_section), pointer :: fine_cumulative_flow_field => null()
        integer :: scale_factor
        contains
            private
            ! In lieu of a final routine as this feature is not currently (August 2016)
            ! supported by all fortran compilers
            ! final :: destructor
            procedure, public :: destructor
            procedure, public :: break_loops
            procedure :: break_loop
            procedure :: find_highest_cumulative_flow_of_cell
            procedure(find_cells_in_loop), deferred :: find_cells_in_loop
            procedure(locate_highest_cumulative_flow_of_cell), deferred :: &
                locate_highest_cumulative_flow_of_cell
            procedure(assign_rdir_of_highest_cumulative_flow_of_cell), deferred :: &
                assign_rdir_of_highest_cumulative_flow_of_cell
            procedure(generate_permitted_rdir), deferred :: generate_permitted_rdir
            procedure(get_no_data_rdir_value), deferred :: get_no_data_rdir_value
    end type loop_breaker

    abstract interface
        function find_cells_in_loop(this,loop_num) result(cells_in_loop)
            import loop_breaker
            import doubly_linked_list
            class(loop_breaker) :: this
            type(doubly_linked_list), pointer :: cells_in_loop
            integer :: loop_num
        end function find_cells_in_loop

        function locate_highest_cumulative_flow_of_cell(this,coords_in, &
            permitted_diagonal_outflow_rdir,vertical_boundary_outflow) result(highest_cumulative_flow_location)
            import loop_breaker
            import coords
            import direction_indicator
            class(loop_breaker) :: this
            class(coords), allocatable :: coords_in
            class(coords), pointer :: highest_cumulative_flow_location
            class(direction_indicator), intent(out), allocatable, optional :: &
                permitted_diagonal_outflow_rdir
            logical, intent(out), optional :: vertical_boundary_outflow
        end function locate_highest_cumulative_flow_of_cell

        subroutine assign_rdir_of_highest_cumulative_flow_of_cell(this,coords_in)
            import loop_breaker
            import coords
            class(loop_breaker) :: this
            class(coords) :: coords_in
        end subroutine assign_rdir_of_highest_cumulative_flow_of_cell

        function generate_permitted_rdir(this,coords_in,section_coords_in) &
            result(permitted_rdir)
            import loop_breaker
            import coords
            import section_coords
            import direction_indicator
            class(loop_breaker) :: this
            class(coords) :: coords_in
            class(section_coords) :: section_coords_in
            class(direction_indicator),pointer :: permitted_rdir
        end function generate_permitted_rdir

        function get_no_data_rdir_value(this) result(no_data_rdir_value)
            import direction_indicator
            import loop_breaker
            class(loop_breaker) :: this
            class(direction_indicator), pointer :: no_data_rdir_value
        end function

    end interface

    type, extends(loop_breaker), abstract, public :: latlon_loop_breaker
            integer :: nlat_course
            integer :: nlon_course
        contains
            private
            procedure :: init_latlon_loop_breaker
            procedure :: find_cells_in_loop => latlon_find_cells_in_loop
            procedure :: locate_highest_cumulative_flow_of_cell => &
                latlon_locate_highest_cumulative_flow_of_cell
            procedure, public :: latlon_get_loop_free_rdirs
    end type latlon_loop_breaker

    type, extends(latlon_loop_breaker), public :: &
        latlon_dir_based_rdirs_loop_breaker
        integer :: no_data_rdir = -999
        contains
            private
            procedure :: assign_rdir_of_highest_cumulative_flow_of_cell => &
                dir_based_rdirs_assign_rdir_of_highest_cumulative_flow_of_cell
            procedure :: generate_permitted_rdir => dir_based_rdirs_generate_permitted_rdir
            procedure :: get_no_data_rdir_value => dir_based_rdirs_get_no_data_rdir_value
    end type latlon_dir_based_rdirs_loop_breaker

    interface latlon_dir_based_rdirs_loop_breaker
        procedure latlon_dir_based_rdirs_loop_breaker_constructor
    end interface latlon_dir_based_rdirs_loop_breaker

contains

    subroutine break_loops(this,loop_nums)
        class(loop_breaker) :: this
        integer, dimension(:) :: loop_nums
        integer :: i
            do i=1,size(loop_nums)
                call this%break_loop(loop_nums(i))
            end do
    end subroutine break_loops

    subroutine destructor(this)
    class(loop_breaker) :: this
        if (associated(this%course_catchment_field)) &
            deallocate(this%course_catchment_field)
        if (associated(this%course_cumulative_flow_field)) &
            deallocate(this%course_cumulative_flow_field)
        if  (associated(this%course_rdirs_field)) &
            deallocate(this%course_rdirs_field)
        if  (associated(this%fine_rdirs_field)) &
            deallocate(this%fine_rdirs_field)
        if  (associated(this%fine_cumulative_flow_field)) &
            deallocate(this%fine_cumulative_flow_field)
    end subroutine destructor

    subroutine break_loop(this,loop_num)
        class(loop_breaker) :: this
        integer :: loop_num
        integer :: current_highest_cumulative_flow
        type(doubly_linked_list), pointer :: cells_in_loop
        class(coords),allocatable :: cell_coords
        class(coords),allocatable :: exit_coords
            cells_in_loop => this%find_cells_in_loop(loop_num)
            if (cells_in_loop%get_length() <= 0) then
                call cells_in_loop%destructor()
                return
            end if
            current_highest_cumulative_flow = 0
            do
                if (cells_in_loop%iterate_forward()) exit
                select type (cell_coords_value => cells_in_loop%get_value_at_iterator_position())
                class is (coords)
                    if (allocated(cell_coords)) deallocate(cell_coords)
                    allocate(cell_coords,source=cell_coords_value)
                end select
                if (this%find_highest_cumulative_flow_of_cell(cell_coords)> &
                    current_highest_cumulative_flow) then
                    current_highest_cumulative_flow = &
                        this%find_highest_cumulative_flow_of_cell(cell_coords)
                    if (allocated(exit_coords)) deallocate(exit_coords)
                    allocate(exit_coords,source=cell_coords)
                end if
            end do
            call this%assign_rdir_of_highest_cumulative_flow_of_cell(exit_coords)
            deallocate(cell_coords)
            deallocate(exit_coords)
            call cells_in_loop%destructor()
            deallocate(cells_in_loop)
    end subroutine break_loop

    function find_highest_cumulative_flow_of_cell(this,coords_in) &
        result (highest_cumulative_flow)
        class(loop_breaker) :: this
        class(coords), allocatable :: coords_in
        class(coords), pointer :: highest_cumulative_flow_location ! high res coords
        class(*), pointer :: highest_cumulative_flow_value
        integer :: highest_cumulative_flow
            highest_cumulative_flow_location => &
                this%locate_highest_cumulative_flow_of_cell(coords_in)
            highest_cumulative_flow_value => &
                this%fine_cumulative_flow_field%get_value(highest_cumulative_flow_location)
            select type(highest_cumulative_flow_value)
            type is (integer)
                  highest_cumulative_flow = highest_cumulative_flow_value
            end select
            deallocate(highest_cumulative_flow_value)
            deallocate(highest_cumulative_flow_location)
    end function

    subroutine init_latlon_loop_breaker(this,course_catchments,course_cumulative_flow,course_rdirs,&
                                        fine_rdirs,fine_cumulative_flow)
        class(latlon_loop_breaker) :: this
        class(*), dimension(:,:), pointer :: course_catchments
        class(*), dimension(:,:), pointer :: course_cumulative_flow
        class(*), dimension(:,:), pointer :: course_rdirs
        class(*), dimension(:,:), pointer :: fine_rdirs
        class(*), dimension(:,:), pointer :: fine_cumulative_flow
        this%course_catchment_field => latlon_field_section(course_catchments,&
            latlon_section_coords(1,1,size(course_catchments,1),size(course_catchments,2)))
        this%course_cumulative_flow_field => latlon_field_section(course_cumulative_flow,&
            latlon_section_coords(1,1,size(course_cumulative_flow,1),size(course_cumulative_flow,2)))
        this%course_rdirs_field => latlon_field_section(course_rdirs, &
            latlon_section_coords(1,1,size(course_rdirs,1),size(course_rdirs,2)))
        this%fine_rdirs_field => latlon_field_section(fine_rdirs, &
            latlon_section_coords(1,1,size(fine_rdirs,1),size(fine_rdirs,2)))
        this%fine_cumulative_flow_field => latlon_field_section(fine_cumulative_flow, &
            latlon_section_coords(1,1,size(fine_cumulative_flow,1),size(fine_cumulative_flow,2)))
        this%scale_factor = size(fine_rdirs,1)/size(course_rdirs,1)
        this%nlat_course = size(course_rdirs,1)
        this%nlon_course = size(course_rdirs,2)
    end subroutine init_latlon_loop_breaker

    function latlon_get_loop_free_rdirs(this) result(loop_free_course_rdirs)
        class(latlon_loop_breaker) :: this
        class(*), dimension(:,:), pointer :: loop_free_course_rdirs
            select type (course_rdirs_field => this%course_rdirs_field)
            class is (latlon_field_section)
                loop_free_course_rdirs => course_rdirs_field%get_data()
            end select
    end function latlon_get_loop_free_rdirs

    function latlon_find_cells_in_loop(this,loop_num) result(cells_in_loop)
        class(latlon_loop_breaker) :: this
        type(doubly_linked_list), pointer :: cells_in_loop
        class(*), pointer :: cumulative_flow_value
        class(*), pointer :: catchment_value
        integer :: loop_num
        integer :: i,j
            allocate(cells_in_loop)
            do j = 1,this%nlon_course
                do i = 1,this%nlat_course
                    cumulative_flow_value => &
                        this%course_cumulative_flow_field%get_value(latlon_coords(i,j))
                    select type (cumulative_flow_value)
                        type is (integer)
                        catchment_value => &
                            this%course_catchment_field%get_value(latlon_coords(i,j))
                        select type (catchment_value)
                        type is (integer)
                            if (catchment_value == loop_num .and. cumulative_flow_value == 0 ) then
                                call cells_in_loop%add_value_to_back(latlon_coords(i,j))
                            end if
                        end select
                        deallocate(catchment_value)
                    end select
                    deallocate(cumulative_flow_value)
                end do
            end do
    end function latlon_find_cells_in_loop

    function latlon_locate_highest_cumulative_flow_of_cell(this,coords_in, &
            permitted_diagonal_outflow_rdir,vertical_boundary_outflow) result(highest_cumulative_flow_location)
        class(latlon_loop_breaker) :: this
        class(coords), allocatable :: coords_in
        class(coords), pointer :: highest_cumulative_flow_location
        class(direction_indicator), intent(out), pointer, optional :: permitted_diagonal_outflow_rdir
        class(*), pointer :: pixel_cumulative_flow
        logical, intent(out), optional :: vertical_boundary_outflow
        type(latlon_section_coords) :: fine_resolution_section_coords
        integer :: fine_resolution_min_lat
        integer :: fine_resolution_max_lat
        integer :: fine_resolution_min_lon
        integer :: fine_resolution_max_lon
        integer :: i,j
        integer :: current_highest_cumulative_flow
            nullify(highest_cumulative_flow_location)
            if (present(permitted_diagonal_outflow_rdir)) nullify(permitted_diagonal_outflow_rdir)
            current_highest_cumulative_flow = 0
            if (present(permitted_diagonal_outflow_rdir)) then
                permitted_diagonal_outflow_rdir => this%get_no_data_rdir_value()
            end if
            if (present(vertical_boundary_outflow)) vertical_boundary_outflow = .False.
            select type (coords_in)
            type is (latlon_coords)
                fine_resolution_min_lat = 1 + this%scale_factor*(coords_in%lat-1)
                fine_resolution_max_lat = this%scale_factor*coords_in%lat
                fine_resolution_min_lon = 1 + this%scale_factor*(coords_in%lon-1)
                fine_resolution_max_lon = this%scale_factor*coords_in%lon
                fine_resolution_section_coords = latlon_section_coords(fine_resolution_min_lat,fine_resolution_min_lon, &
                                                    fine_resolution_max_lat - fine_resolution_min_lat + 1, &
                                                    fine_resolution_max_lon - fine_resolution_min_lon + 1)
            end select
            do j = fine_resolution_min_lon,fine_resolution_max_lon
                do i = fine_resolution_min_lat,fine_resolution_max_lat
                    pixel_cumulative_flow => &
                        this%fine_cumulative_flow_field%get_value(latlon_coords(i,j))
                    select type(pixel_cumulative_flow)
                        type is (integer)
                        if ( pixel_cumulative_flow > &
                            current_highest_cumulative_flow) then
                            current_highest_cumulative_flow = pixel_cumulative_flow
                            if (associated(highest_cumulative_flow_location)) deallocate(highest_cumulative_flow_location)
                            allocate(highest_cumulative_flow_location,source=latlon_coords(i,j))
                            if (present(vertical_boundary_outflow)) then
                                if ( j == fine_resolution_min_lon .or. j == fine_resolution_max_lon ) then
                                    vertical_boundary_outflow = .True.
                                else
                                    vertical_boundary_outflow = .False.
                                end if
                            end if
                            if (present(vertical_boundary_outflow) .and. present(permitted_diagonal_outflow_rdir)) then
                                if ( (i == fine_resolution_min_lat .or. i == fine_resolution_max_lat) .and. &
                                    vertical_boundary_outflow) then
                                    if (associated(permitted_diagonal_outflow_rdir)) &
                                        deallocate(permitted_diagonal_outflow_rdir)
                                        permitted_diagonal_outflow_rdir => &
                                            this%generate_permitted_rdir(latlon_coords(i,j), &
                                                fine_resolution_section_coords)
                                else
                                    if (associated(permitted_diagonal_outflow_rdir)) &
                                        deallocate(permitted_diagonal_outflow_rdir)
                                        permitted_diagonal_outflow_rdir => this%get_no_data_rdir_value()
                                end if
                            end if
                        end if
                    end select
                    deallocate(pixel_cumulative_flow)
                end do
            end do
    end function latlon_locate_highest_cumulative_flow_of_cell


    function latlon_dir_based_rdirs_loop_breaker_constructor(course_catchments_in,course_cumulative_flow_in,course_rdirs_in,&
                                                             fine_rdirs_in,fine_cumulative_flow_in) result(constructor)
        type(latlon_dir_based_rdirs_loop_breaker) :: constructor
        integer, dimension(:,:), pointer :: course_catchments_in
        integer, dimension(:,:), pointer :: course_cumulative_flow_in
        integer, dimension(:,:), pointer :: course_rdirs_in
        integer, dimension(:,:), pointer :: fine_cumulative_flow_in
        integer, dimension(:,:), pointer :: fine_rdirs_in
        class(*), dimension(:,:), pointer :: course_catchments
        class(*), dimension(:,:), pointer :: course_cumulative_flow
        class(*), dimension(:,:), pointer :: course_rdirs
        class(*), dimension(:,:), pointer :: fine_cumulative_flow
        class(*), dimension(:,:), pointer :: fine_rdirs
            course_catchments => course_catchments_in
            course_cumulative_flow => course_cumulative_flow_in
            course_rdirs => course_rdirs_in
            fine_cumulative_flow => fine_cumulative_flow_in
            fine_rdirs => fine_rdirs_in
            call constructor%init_latlon_loop_breaker(course_catchments,course_cumulative_flow,course_rdirs,&
                                                      fine_rdirs,fine_cumulative_flow)
    end function latlon_dir_based_rdirs_loop_breaker_constructor

    subroutine dir_based_rdirs_assign_rdir_of_highest_cumulative_flow_of_cell(&
        this,coords_in)
        class(latlon_dir_based_rdirs_loop_breaker) :: this
        class(coords), allocatable :: coords_in
        class(coords), pointer :: highest_cumulative_flow_location
        class(direction_indicator), pointer :: permitted_diagonal_outflow_rdir
        class(*), pointer :: rdir
        integer, dimension(4) :: corner_rdirs
        logical :: vertical_boundary_outflow
            corner_rdirs = (/ 7,9,1,3 /)
            highest_cumulative_flow_location => &
                this%locate_highest_cumulative_flow_of_cell(coords_in, &
                    permitted_diagonal_outflow_rdir,vertical_boundary_outflow)
            rdir => this%fine_rdirs_field%get_value(highest_cumulative_flow_location)
            deallocate(highest_cumulative_flow_location)
            select type (rdir)
            type is (integer)
                ! use of a generic interface to is_equal_to not possible in this case
                if (any(rdir == corner_rdirs) .and. &
                    .not. permitted_diagonal_outflow_rdir%is_equal_to_integer(rdir)) then
                    if (.not. permitted_diagonal_outflow_rdir%is_equal_to_integer(this%no_data_rdir) ) then
                        if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(7) ) then
                            if (rdir == 1) then
                                call this%course_rdirs_field%set_value(coords_in,4)
                            else
                                call this%course_rdirs_field%set_value(coords_in,8)
                            end if
                        else if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(9) ) then
                            if (rdir == 7) then
                                call this%course_rdirs_field%set_value(coords_in,8)
                            else
                                call this%course_rdirs_field%set_value(coords_in,6)
                            end if
                        else if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(3) ) then
                            if (rdir == 9) then
                                call this%course_rdirs_field%set_value(coords_in,6)
                            else
                                call this%course_rdirs_field%set_value(coords_in,2)
                            end if
                        else if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(1) ) then
                            if (rdir == 3) then
                                call this%course_rdirs_field%set_value(coords_in,2)
                            else
                                call this%course_rdirs_field%set_value(coords_in,4)
                            end if
                        end if
                    else if (vertical_boundary_outflow) then
                        if (rdir == 7 .or. rdir == 1) then
                            call this%course_rdirs_field%set_value(coords_in,4)
                        else
                            call this%course_rdirs_field%set_value(coords_in,6)
                        end if
                    else
                         if (rdir == 7 .or. rdir == 9) then
                            call this%course_rdirs_field%set_value(coords_in,8)
                         else
                            call this%course_rdirs_field%set_value(coords_in,2)
                         end if
                    end if
                else
                    call this%course_rdirs_field%set_value(coords_in,rdir)
                end if
            end select
            deallocate(permitted_diagonal_outflow_rdir)
            deallocate(rdir)
    end subroutine dir_based_rdirs_assign_rdir_of_highest_cumulative_flow_of_cell

    function dir_based_rdirs_generate_permitted_rdir(this,coords_in,section_coords_in) result(permitted_rdir)
        class(latlon_dir_based_rdirs_loop_breaker) :: this
        class(coords) :: coords_in
        class(section_coords) :: section_coords_in
        class(direction_indicator), pointer :: permitted_rdir
        integer :: section_max_lat,section_max_lon
            select type (coords_in)
            type is (latlon_coords)
                select type(section_coords_in)
                type is (latlon_section_coords)
                    section_max_lat = section_coords_in%section_min_lat + section_coords_in%section_width_lat - 1
                    section_max_lon = section_coords_in%section_min_lon + section_coords_in%section_width_lon - 1
                    if (coords_in%lat == section_coords_in%section_min_lat .and. &
                        coords_in%lon == section_coords_in%section_min_lon) then
                        allocate(permitted_rdir,source=dir_based_direction_indicator(7))
                    else if (coords_in%lat == section_coords_in%section_min_lat .and. &
                        coords_in%lon == section_max_lon) then
                        allocate(permitted_rdir,source=dir_based_direction_indicator(9))
                    else if (coords_in%lat == section_max_lat .and. &
                        coords_in%lon == section_coords_in%section_min_lon) then
                        allocate(permitted_rdir,source=dir_based_direction_indicator(1))
                    else if (coords_in%lat == section_max_lat .and. &
                        coords_in%lon == section_max_lon) then
                        allocate(permitted_rdir,source=dir_based_direction_indicator(3))
                    else
                        allocate(permitted_rdir,source=dir_based_direction_indicator(this%no_data_rdir))
                    end if
                end select
            end select
    end function dir_based_rdirs_generate_permitted_rdir

    function dir_based_rdirs_get_no_data_rdir_value(this) result(no_data_rdir_value)
        class(latlon_dir_based_rdirs_loop_breaker) :: this
        class(direction_indicator), pointer :: no_data_rdir_value
            allocate(no_data_rdir_value,source=dir_based_direction_indicator(this%no_data_rdir))
    end function dir_based_rdirs_get_no_data_rdir_value

end module loop_breaker_mod
