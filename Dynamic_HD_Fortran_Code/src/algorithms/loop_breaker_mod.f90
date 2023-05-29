module loop_breaker_mod
    use coords_mod
    use field_section_mod
    use doubly_linked_list_mod
    implicit none
    private

    !> An abstract class containing a loop breaker (i.e. tool to remove upwanted
    !! loops in upscaled river directions) for a generic grid
    type, abstract, public :: loop_breaker
        private
        !> The coarse catchments as a field section covering the entire field
        class(field_section), pointer :: coarse_catchment_field => null()
        !> The coarse total cumulative flow as a field section covering the entire field
        class(field_section), pointer :: coarse_cumulative_flow_field => null()
        !> The coarse river directions as a field section covering the entire field
        class(field_section), pointer :: coarse_rdirs_field => null()
        !> The fine river directions as a field section covering the entire field
        class(field_section), pointer :: fine_rdirs_field => null()
        !> The fine total cumulative flow as a field section covering the entire field
        class(field_section), pointer :: fine_cumulative_flow_field => null()
        !> The number of fine cells for every coarse cell
        integer :: scale_factor
        contains
            private
            ! In lieu of a final routine as this feature is not currently (August 2016)
            ! supported by all fortran compilers
            ! final :: destructor
            !> Destructor; delete any arrays necessary
            procedure, public :: destructor
            !> Break each loop in the input list of loops
            procedure, public :: break_loops
            !> Break the loop with the loop number specified as an argument
            procedure :: break_loop
            !> Find the value of the highest cumulative flow in a cell at a given
            !! set of coarse coordinates and return it as an integer
            procedure :: find_highest_cumulative_flow_of_cell
            !> Find the set of cell downstream of a given cell in the current coarse river
            !! directions tracing the path all the way to an outflow point
            procedure :: find_cells_downstream
            !> Find which cells are in a given loop; input is the loop number of
            !! desired loop; output a list of the coarse coordinates of the cells
            !! in the loop
            procedure(find_cells_in_loop), deferred :: find_cells_in_loop
            !> Locate the fine cell with the highest cumulative flow within the cell and
            !! return its coordinates. Input is a set of coarse coordinates for the
            !! cell; output is a set of fine coordinates for the fine cell with the highest
            !! cumulative flow. Also can optionally return a flag indicating if this
            !! is next to a vertical edge (hence the flow likely crosses that edge) and a
            !! direction indicator that is either set to the no data value (if this is not
            !! a corner pixel/fine cell) or a direction indicating which diagonal flow
            !! direction would be possible from this cell.
            procedure(locate_highest_cumulative_flow_of_cell), deferred :: &
                locate_highest_cumulative_flow_of_cell
            !> Set the rdir of a coarse cell based on the rdir of the highest cumulative
            !! flow fine cell. At non-corner cells simply the choose the correct direction
            !! to cross the edge the cell is next to. For corner cells choose the cell that
            !! the fine river direction of the cell with highest cumulative flow points
            !! towards. Takes as input the fine coordinates of the highest cumulative flow
            !! cell and return nothing directly (but changes the state of this).
            procedure(assign_rdir_of_highest_cumulative_flow_of_cell), deferred :: &
                assign_rdir_of_highest_cumulative_flow_of_cell
            !> Function that takes the fine coordinates of the cell with the highest
            !! cumulative flow along with the fine section coordinates that describe
            !! the area that maps to the coarse cell and return the permitted diagonal
            !! river direction (the one pointing out diagonally from the corner between
            !! the vertical and horizontal directions possible from the cell with
            !! the highest cumulative flow) as a direction indicator
            procedure(generate_permitted_rdir), deferred :: generate_permitted_rdir
            !> Return a suitable direction indicator to represent a no data flow
            !! direction.
            procedure(get_no_data_rdir_value), deferred :: get_no_data_rdir_value
            procedure(find_next_cell_downstream), deferred :: find_next_cell_downstream
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
            class(direction_indicator), intent(out), pointer, optional :: &
                permitted_diagonal_outflow_rdir
            logical, intent(out), optional :: vertical_boundary_outflow
        end function locate_highest_cumulative_flow_of_cell

        subroutine assign_rdir_of_highest_cumulative_flow_of_cell(this,coords_in)
            import loop_breaker
            import coords
            class(loop_breaker) :: this
            class(coords), allocatable :: coords_in
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
        end function get_no_data_rdir_value

        subroutine find_next_cell_downstream(this,coords_inout,&
                                             outflow_cell_reached)
            import loop_breaker
            import coords
            class(loop_breaker) :: this
            class(coords), intent(inout) :: coords_inout
            logical, intent(out) :: outflow_cell_reached
        end subroutine find_next_cell_downstream

    end interface

    !> Abstract subclass of loop_breaker for a latitude longitude grid
    type, extends(loop_breaker), abstract, public :: latlon_loop_breaker
            !> The number of coarse latitude points
            integer :: nlat_coarse
            !> The number of coarse longitude points
            integer :: nlon_coarse
        contains
            private
            !> Subroutine to initialise a latitude longitude loop breaker. Input
            !! arguements are a set of coarse catchments, the coarse cumulative
            !! flow field and the coarse river directions along with the fine river
            !! directions and fine cumulative flow field
            procedure :: init_latlon_loop_breaker
            !> Find which cells are in a given loop; input is the loop number of
            !! desired loop; output a list of the coarse coordinates of the cells
            !! in the loop
            procedure :: find_cells_in_loop => latlon_find_cells_in_loop
            !> Locate the fine cell with the highest cumulative flow within the cell and
            !! return its coordinates. Input is a set of coarse coordinates for the
            !! cell; output is a set of fine coordinates for the fine cell with the highest
            !! cumulative flow. Also can optionally return a flag indicating if this
            !! is next to a vertical edge (hence the flow likely crosses that edge) and a
            !! direction indicator that is either set to the no data value (if this is not
            !! a corner pixel/fine cell) or a direction indicating which diagonal flow
            !! direction would be possible from this cell.
            procedure :: locate_highest_cumulative_flow_of_cell => &
                latlon_locate_highest_cumulative_flow_of_cell
            !> Return the a pointer to the loop free coarse river directions produced; this
            !! is to be run after the loop breaker has been run in order to retrieve the
            !! results
            procedure, public :: latlon_get_loop_free_rdirs
    end type latlon_loop_breaker

    type, extends(loop_breaker), abstract, public :: &
        icon_icosohedral_cell_latlon_pixel_loop_breaker
            integer :: ncells
            integer, dimension(:,:), pointer :: expanded_cell_numbers_data
            integer, dimension(:), pointer :: section_min_lats
            integer, dimension(:), pointer :: section_min_lons
            integer, dimension(:), pointer :: section_max_lats
            integer, dimension(:), pointer :: section_max_lons
            integer :: index_for_sink = -5
            integer :: index_for_outflow = -1
            integer :: index_for_ocean_point = -2
        contains
            private
            procedure :: init_icon_icosohedral_cell_latlon_pixel_loop_breaker
            procedure :: find_cells_in_loop => icon_icosohedral_cell_latlon_pixel_find_cells_in_loop
            procedure :: locate_highest_cumulative_flow_of_cell => &
                icon_i_c_latlon_p_locate_highest_cumulative_flow_of_cell
            procedure, public :: icon_icosohedral_cell_get_loop_free_rdirs
    end type icon_icosohedral_cell_latlon_pixel_loop_breaker

    !> Concrete subclass of a latitude longitude loop breaker for direction (1-9 keypad code)
    !! based river directions
    type, extends(latlon_loop_breaker), public :: &
        latlon_dir_based_rdirs_loop_breaker
        !> Code for a no data river direction
        integer :: no_data_rdir = -999
        contains
            private
            !> Set the rdir of a coarse cell based on the rdir of the highest cumulative
            !! flow fine cell. At non-corner cells simply the choose the correct direction
            !! to cross the edge the cell is next to. For corner cells choose the cell that
            !! the fine river direction of the cell with highest cumulative flow points
            !! towards. Takes as input the fine coordinates of the highest cumulative flow
            !! cell and return nothing directly (but changes the state of this).
            procedure :: assign_rdir_of_highest_cumulative_flow_of_cell => &
                dir_based_rdirs_assign_rdir_of_highest_cumulative_flow_of_cell
            !> Function that takes the fine coordinates of the cell with the highest
            !! cumulative flow along with the fine section coordinates that describe
            !! the area that maps to the coarse cell and return the permitted diagonal
            !! river direction (the one pointing out diagonally from the corner between
            !! the vertical and horizontal directions possible from the cell with
            !! the highest cumulative flow) as a direction based direction indicator
            procedure :: generate_permitted_rdir => dir_based_rdirs_generate_permitted_rdir
            !> Return a direction based direction indicator to represent a no data flow
            !! direction.
            procedure :: get_no_data_rdir_value => dir_based_rdirs_get_no_data_rdir_value
            procedure :: find_next_cell_downstream => &
                         latlon_dir_based_rdirs_find_next_cell_downstream
    end type latlon_dir_based_rdirs_loop_breaker

    interface latlon_dir_based_rdirs_loop_breaker
        procedure :: latlon_dir_based_rdirs_loop_breaker_constructor
    end interface latlon_dir_based_rdirs_loop_breaker

    !> Concrete subclass of a latitude longitude loop breaker for direction (1-9 keypad code)
    !! based river directions
    type, extends(icon_icosohedral_cell_latlon_pixel_loop_breaker), public :: &
        icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker
        !> Code for a no data river direction
        integer :: no_data_rdir = -999
        contains
            private
            !> Set the rdir of a coarse cell based on the rdir of the highest cumulative
            !! flow fine cell. At non-corner cells simply the choose the correct direction
            !! to cross the edge the cell is next to. For corner cells choose the cell that
            !! the fine river direction of the cell with highest cumulative flow points
            !! towards. Takes as input the fine coordinates of the highest cumulative flow
            !! cell and return nothing directly (but changes the state of this).
            procedure :: assign_rdir_of_highest_cumulative_flow_of_cell => &
                irregular_dbr_assign_rdir_of_highest_cumulative_flow_of_cell
            !> Function that takes the fine coordinates of the cell with the highest
            !! cumulative flow along with the fine section coordinates that describe
            !! the area that maps to the coarse cell and return the permitted diagonal
            !! river direction (the one pointing out diagonally from the corner between
            !! the vertical and horizontal directions possible from the cell with
            !! the highest cumulative flow) as a direction based direction indicator
            procedure :: generate_permitted_rdir => irregular_latlon_dir_based_rdirs_generate_permitted_rdir
            !> Return a direction based direction indicator to represent a no data flow
            !! direction.
            procedure :: get_no_data_rdir_value => irregular_dir_based_rdirs_get_no_data_rdir_value
            procedure :: iic_llp_dir_based_rdirs_calculate_new_coarse_coords
            procedure :: find_next_cell_downstream => &
                         icon_icosohedral_cell_find_next_cell_downstream
    end type icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker

    interface icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker
        procedure :: iic_llp_dir_based_rdirs_loop_breaker_constructor
    end interface icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker

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
        if (associated(this%coarse_catchment_field)) &
            deallocate(this%coarse_catchment_field)
        if (associated(this%coarse_cumulative_flow_field)) &
            deallocate(this%coarse_cumulative_flow_field)
        if  (associated(this%coarse_rdirs_field)) &
            deallocate(this%coarse_rdirs_field)
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
        type(doubly_linked_list), pointer :: downstream_cells
        class(coords),allocatable :: cell_coords
        class(coords),allocatable :: exit_coords
        class(*), pointer :: catchment_value
        logical :: new_path_reenters_catchment_downstream
        logical :: outflow_cell_reached
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
            new_path_reenters_catchment_downstream = .false.
            downstream_cells => this%find_cells_downstream(exit_coords)
            do
                if (downstream_cells%iterate_forward()) exit
                select type (cell_coords_value => downstream_cells%get_value_at_iterator_position())
                class is (coords)
                    if (allocated(cell_coords)) deallocate(cell_coords)
                    allocate(cell_coords,source=cell_coords_value)
                end select
                catchment_value => &
                    this%coarse_catchment_field%get_value(cell_coords)
                select type (catchment_value)
                type is (integer)
                    if (catchment_value == loop_num) then
                        new_path_reenters_catchment_downstream = .true.
                    end if
                end select
                deallocate(catchment_value)
            end do
            if (new_path_reenters_catchment_downstream) then
                allocate(cell_coords,source=exit_coords)
                do
                    call this%find_next_cell_downstream(cell_coords,&
                                                        outflow_cell_reached)
                    if (outflow_cell_reached) exit
                    call this%assign_rdir_of_highest_cumulative_flow_of_cell(cell_coords)
                end do
            end if
            deallocate(cell_coords)
            deallocate(exit_coords)
            call cells_in_loop%destructor()
            call downstream_cells%destructor()
            deallocate(cells_in_loop)
            deallocate(downstream_cells)
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

    function find_cells_downstream(this,coords_in) result(cells_downstream)
        class(loop_breaker) :: this
        class(coords), allocatable :: coords_in
        class(coords), allocatable :: cell_coords
        type(doubly_linked_list), pointer :: cells_downstream
        logical :: outflow_cell_reached
            allocate(cells_downstream)
            allocate(cell_coords,source=coords_in)
            do
                call this%find_next_cell_downstream(cell_coords,&
                                                    outflow_cell_reached)
                if (outflow_cell_reached) exit
                call cells_downstream%add_value_to_back(cell_coords)
            end do
    end function find_cells_downstream

    subroutine init_latlon_loop_breaker(this,coarse_catchments,coarse_cumulative_flow,coarse_rdirs,&
                                        fine_rdirs,fine_cumulative_flow)
        class(latlon_loop_breaker) :: this
        class(*), dimension(:,:), pointer :: coarse_catchments
        class(*), dimension(:,:), pointer :: coarse_cumulative_flow
        class(*), dimension(:,:), pointer :: coarse_rdirs
        class(*), dimension(:,:), pointer :: fine_rdirs
        class(*), dimension(:,:), pointer :: fine_cumulative_flow
            this%coarse_catchment_field => latlon_field_section(coarse_catchments,&
                latlon_section_coords(1,1,size(coarse_catchments,1),size(coarse_catchments,2)))
            this%coarse_cumulative_flow_field => latlon_field_section(coarse_cumulative_flow,&
                latlon_section_coords(1,1,size(coarse_cumulative_flow,1),size(coarse_cumulative_flow,2)))
            this%coarse_rdirs_field => latlon_field_section(coarse_rdirs, &
                latlon_section_coords(1,1,size(coarse_rdirs,1),size(coarse_rdirs,2)))
            this%fine_rdirs_field => latlon_field_section(fine_rdirs, &
                latlon_section_coords(1,1,size(fine_rdirs,1),size(fine_rdirs,2)))
            this%fine_cumulative_flow_field => latlon_field_section(fine_cumulative_flow, &
                latlon_section_coords(1,1,size(fine_cumulative_flow,1),size(fine_cumulative_flow,2)))
            this%scale_factor = size(fine_rdirs,1)/size(coarse_rdirs,1)
            this%nlat_coarse = size(coarse_rdirs,1)
            this%nlon_coarse = size(coarse_rdirs,2)
    end subroutine init_latlon_loop_breaker

    subroutine init_icon_icosohedral_cell_latlon_pixel_loop_breaker(this,&
                                                                    coarse_catchments,&
                                                                    coarse_cumulative_flow,&
                                                                    coarse_rdirs,&
                                                                    coarse_section_coords,&
                                                                    fine_rdirs,&
                                                                    fine_cumulative_flow, &
                                                                    expanded_cell_numbers_data, &
                                                                    section_min_lats, &
                                                                    section_min_lons, &
                                                                    section_max_lats, &
                                                                    section_max_lons)
        class(icon_icosohedral_cell_latlon_pixel_loop_breaker) :: this
        class(*), dimension(:), pointer :: coarse_catchments
        class(*), dimension(:), pointer :: coarse_cumulative_flow
        type(generic_1d_section_coords) :: coarse_section_coords
        class(*), dimension(:), pointer :: coarse_rdirs
        class(*), dimension(:,:), pointer :: fine_rdirs
        class(*), dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expanded_cell_numbers_data
        integer, dimension(:), pointer :: section_min_lats
        integer, dimension(:), pointer :: section_min_lons
        integer, dimension(:), pointer :: section_max_lats
        integer, dimension(:), pointer :: section_max_lons
            this%coarse_catchment_field => icon_single_index_field_section(coarse_catchments,&
                                                                           coarse_section_coords)
            this%coarse_cumulative_flow_field => icon_single_index_field_section(coarse_cumulative_flow,&
                                                                                 coarse_section_coords)
            this%coarse_rdirs_field => icon_single_index_field_section(coarse_rdirs,&
                                                                       coarse_section_coords)
            this%fine_rdirs_field => latlon_field_section(fine_rdirs, &
                latlon_section_coords(1,1,size(fine_rdirs,1),size(fine_rdirs,2)))
            this%fine_cumulative_flow_field => latlon_field_section(fine_cumulative_flow, &
                latlon_section_coords(1,1,size(fine_cumulative_flow,1),size(fine_cumulative_flow,2)))
            this%ncells = size(coarse_rdirs)
            this%expanded_cell_numbers_data => expanded_cell_numbers_data
            this%section_min_lats => section_min_lats
            this%section_min_lons => section_min_lons
            this%section_max_lats => section_max_lats
            this%section_max_lons => section_max_lons
    end subroutine init_icon_icosohedral_cell_latlon_pixel_loop_breaker

    function latlon_get_loop_free_rdirs(this) result(loop_free_coarse_rdirs)
        class(latlon_loop_breaker) :: this
        class(*), dimension(:,:), pointer :: loop_free_coarse_rdirs
            select type (coarse_rdirs_field => this%coarse_rdirs_field)
            class is (latlon_field_section)
                loop_free_coarse_rdirs => coarse_rdirs_field%get_data()
            end select
    end function latlon_get_loop_free_rdirs

    function icon_icosohedral_cell_get_loop_free_rdirs(this) result(loop_free_coarse_rdirs)
        class(icon_icosohedral_cell_latlon_pixel_loop_breaker) :: this
        class(*), dimension(:), pointer :: loop_free_coarse_rdirs
            select type (coarse_rdirs_field => this%coarse_rdirs_field)
            class is (icon_single_index_field_section)
                loop_free_coarse_rdirs => coarse_rdirs_field%get_data()
            end select
    end function icon_icosohedral_cell_get_loop_free_rdirs

    function latlon_find_cells_in_loop(this,loop_num) result(cells_in_loop)
        class(latlon_loop_breaker) :: this
        type(doubly_linked_list), pointer :: cells_in_loop
        class(*), pointer :: cumulative_flow_value
        class(*), pointer :: catchment_value
        integer :: loop_num
        integer :: i,j
            allocate(cells_in_loop)
            do j = 1,this%nlon_coarse
                do i = 1,this%nlat_coarse
                    cumulative_flow_value => &
                        this%coarse_cumulative_flow_field%get_value(latlon_coords(i,j))
                    select type (cumulative_flow_value)
                        type is (integer)
                        catchment_value => &
                            this%coarse_catchment_field%get_value(latlon_coords(i,j))
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

    function icon_icosohedral_cell_latlon_pixel_find_cells_in_loop(this,loop_num) result(cells_in_loop)
        class(icon_icosohedral_cell_latlon_pixel_loop_breaker) :: this
        type(doubly_linked_list), pointer :: cells_in_loop
        class(*), pointer :: cumulative_flow_value
        class(*), pointer :: catchment_value
        integer :: loop_num
        integer :: i
            allocate(cells_in_loop)
            do i = 1,this%ncells
                cumulative_flow_value => &
                    this%coarse_cumulative_flow_field%get_value(generic_1d_coords(i))
                select type (cumulative_flow_value)
                    type is (integer)
                    catchment_value => &
                        this%coarse_catchment_field%get_value(generic_1d_coords(i))
                    select type (catchment_value)
                    type is (integer)
                        if (catchment_value == loop_num .and. cumulative_flow_value == 0 ) then
                            call cells_in_loop%add_value_to_back(generic_1d_coords(i))
                        end if
                    end select
                    deallocate(catchment_value)
                end select
                deallocate(cumulative_flow_value)
            end do
    end function icon_icosohedral_cell_latlon_pixel_find_cells_in_loop

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
                                ! If this is the further right or left column then we have a flow across
                                ! the vertical boundary; at least potentially
                                if ( j == fine_resolution_min_lon .or. j == fine_resolution_max_lon ) then
                                    vertical_boundary_outflow = .True.
                                else
                                    vertical_boundary_outflow = .False.
                                end if
                            end if
                            if (present(vertical_boundary_outflow) .and. present(permitted_diagonal_outflow_rdir)) then
                                ! If this is both on the vertical and horizontal boundary it is a corner and
                                ! thus a diagonal outflow in one particular direction is possible
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

    function icon_i_c_latlon_p_locate_highest_cumulative_flow_of_cell(this,coords_in, &
            permitted_diagonal_outflow_rdir,vertical_boundary_outflow) result(highest_cumulative_flow_location)
        class(icon_icosohedral_cell_latlon_pixel_loop_breaker) :: this
        class(coords), allocatable :: coords_in
        class(coords), pointer :: highest_cumulative_flow_location
        class(direction_indicator), intent(out), pointer, optional :: permitted_diagonal_outflow_rdir
        class(*), pointer :: pixel_cumulative_flow
        logical, intent(out), optional :: vertical_boundary_outflow
        type(irregular_latlon_section_coords) :: fine_resolution_section_coords
        integer :: fine_resolution_min_lat
        integer :: fine_resolution_max_lat
        integer :: fine_resolution_min_lon
        integer :: fine_resolution_max_lon
        integer :: cell_number
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
            type is (generic_1d_coords)
                fine_resolution_section_coords = irregular_latlon_section_coords(coords_in%index, &
                                                                                 this%expanded_cell_numbers_data, &
                                                                                 this%section_min_lats, &
                                                                                 this%section_min_lons, &
                                                                                 this%section_max_lats, &
                                                                                 this%section_max_lons,&
                                                                                 1)
            end select
            fine_resolution_min_lat = fine_resolution_section_coords%section_min_lat
            fine_resolution_max_lat = fine_resolution_section_coords%section_min_lat + &
                                      fine_resolution_section_coords%section_width_lat - 1
            fine_resolution_min_lon = fine_resolution_section_coords%section_min_lon
            fine_resolution_max_lon = fine_resolution_section_coords%section_min_lon + &
                                      fine_resolution_section_coords%section_width_lon - 1
            cell_number = fine_resolution_section_coords%cell_number
            do j = fine_resolution_min_lon,fine_resolution_max_lon
                do i = fine_resolution_min_lat,fine_resolution_max_lat
                    if(cell_number == fine_resolution_section_coords%cell_numbers(i,j)) then
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
                                    if ( j == fine_resolution_min_lon .or. j == fine_resolution_max_lon .or. &
                                         (.not. (cell_number == &
                                                 fine_resolution_section_coords%cell_numbers(i,j+1))) .or. &
                                         (.not. (cell_number == &
                                                 fine_resolution_section_coords%cell_numbers(i,j-1)))) then
                                        vertical_boundary_outflow = .True.
                                    else
                                        vertical_boundary_outflow = .False.
                                    end if
                                end if
                                if (present(vertical_boundary_outflow) .and. present(permitted_diagonal_outflow_rdir)) then
                                    ! If this is both on the vertical and horizontal boundary it is a corner and
                                    ! thus a diagonal outflow in one particular direction is possible
                                    if ( (i == fine_resolution_min_lat .or. i == fine_resolution_max_lat .or. &
                                         (.not. (cell_number == &
                                                 fine_resolution_section_coords%cell_numbers(i+1,j))) .or. &
                                         (.not. (cell_number == &
                                                 fine_resolution_section_coords%cell_numbers(i-1,j)))) .and. &
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
                    end if
                end do
            end do
    end function icon_i_c_latlon_p_locate_highest_cumulative_flow_of_cell

    function latlon_dir_based_rdirs_loop_breaker_constructor(coarse_catchments_in,coarse_cumulative_flow_in,coarse_rdirs_in,&
                                                             fine_rdirs_in,fine_cumulative_flow_in) result(constructor)
        type(latlon_dir_based_rdirs_loop_breaker) :: constructor
        integer, dimension(:,:), pointer :: coarse_catchments_in
        integer, dimension(:,:), pointer :: coarse_cumulative_flow_in
        integer, dimension(:,:), pointer :: coarse_rdirs_in
        integer, dimension(:,:), pointer :: fine_cumulative_flow_in
        integer, dimension(:,:), pointer :: fine_rdirs_in
        class(*), dimension(:,:), pointer :: coarse_catchments
        class(*), dimension(:,:), pointer :: coarse_cumulative_flow
        class(*), dimension(:,:), pointer :: coarse_rdirs
        class(*), dimension(:,:), pointer :: fine_cumulative_flow
        class(*), dimension(:,:), pointer :: fine_rdirs
            coarse_catchments => coarse_catchments_in
            coarse_cumulative_flow => coarse_cumulative_flow_in
            coarse_rdirs => coarse_rdirs_in
            fine_cumulative_flow => fine_cumulative_flow_in
            fine_rdirs => fine_rdirs_in
            call constructor%init_latlon_loop_breaker(coarse_catchments,coarse_cumulative_flow,coarse_rdirs,&
                                                      fine_rdirs,fine_cumulative_flow)
    end function latlon_dir_based_rdirs_loop_breaker_constructor

    function iic_llp_dir_based_rdirs_loop_breaker_constructor(coarse_catchments_in,&
                                                              coarse_cumulative_flow_in,&
                                                              coarse_rdirs_in,&
                                                              coarse_section_coords_in, &
                                                              fine_rdirs_in,fine_cumulative_flow_in, &
                                                              expanded_cell_numbers_data, &
                                                              section_min_lats, &
                                                              section_min_lons, &
                                                              section_max_lats, &
                                                              section_max_lons) &
                                                              result(constructor)
        type(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: constructor
        type(generic_1d_section_coords) :: coarse_section_coords_in
        integer, dimension(:), pointer :: coarse_catchments_in
        integer, dimension(:), pointer :: coarse_cumulative_flow_in
        integer, dimension(:), pointer :: coarse_rdirs_in
        integer, dimension(:,:), pointer :: fine_cumulative_flow_in
        integer, dimension(:,:), pointer :: fine_rdirs_in
        class(*), dimension(:), pointer :: coarse_catchments
        class(*), dimension(:), pointer :: coarse_cumulative_flow
        class(*), dimension(:), pointer :: coarse_rdirs
        class(*), dimension(:,:), pointer :: fine_cumulative_flow
        class(*), dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: expanded_cell_numbers_data
        integer, dimension(:), pointer :: section_min_lats
        integer, dimension(:), pointer :: section_min_lons
        integer, dimension(:), pointer :: section_max_lats
        integer, dimension(:), pointer :: section_max_lons
            coarse_catchments => coarse_catchments_in
            coarse_cumulative_flow => coarse_cumulative_flow_in
            coarse_rdirs => coarse_rdirs_in
            fine_cumulative_flow => fine_cumulative_flow_in
            fine_rdirs => fine_rdirs_in
            call constructor%init_icon_icosohedral_cell_latlon_pixel_loop_breaker(coarse_catchments,&
                                                                                  coarse_cumulative_flow,&
                                                                                  coarse_rdirs,&
                                                                                  coarse_section_coords_in,&
                                                                                  fine_rdirs,&
                                                                                  fine_cumulative_flow, &
                                                                                  expanded_cell_numbers_data, &
                                                                                  section_min_lats, &
                                                                                  section_min_lons, &
                                                                                  section_max_lats, &
                                                                                  section_max_lons)
    end function iic_llp_dir_based_rdirs_loop_breaker_constructor

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
                                call this%coarse_rdirs_field%set_value(coords_in,4)
                            else
                                call this%coarse_rdirs_field%set_value(coords_in,8)
                            end if
                        else if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(9) ) then
                            if (rdir == 7) then
                                call this%coarse_rdirs_field%set_value(coords_in,8)
                            else
                                call this%coarse_rdirs_field%set_value(coords_in,6)
                            end if
                        else if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(3) ) then
                            if (rdir == 9) then
                                call this%coarse_rdirs_field%set_value(coords_in,6)
                            else
                                call this%coarse_rdirs_field%set_value(coords_in,2)
                            end if
                        else if ( permitted_diagonal_outflow_rdir%is_equal_to_integer(1) ) then
                            if (rdir == 3) then
                                call this%coarse_rdirs_field%set_value(coords_in,2)
                            else
                                call this%coarse_rdirs_field%set_value(coords_in,4)
                            end if
                        end if
                    else if (vertical_boundary_outflow) then
                        if (rdir == 7 .or. rdir == 1) then
                            call this%coarse_rdirs_field%set_value(coords_in,4)
                        else
                            call this%coarse_rdirs_field%set_value(coords_in,6)
                        end if
                    else
                         if (rdir == 7 .or. rdir == 9) then
                            call this%coarse_rdirs_field%set_value(coords_in,8)
                         else
                            call this%coarse_rdirs_field%set_value(coords_in,2)
                         end if
                    end if
                else
                    call this%coarse_rdirs_field%set_value(coords_in,rdir)
                end if
            end select
            deallocate(permitted_diagonal_outflow_rdir)
            deallocate(rdir)
    end subroutine dir_based_rdirs_assign_rdir_of_highest_cumulative_flow_of_cell

    subroutine irregular_dbr_assign_rdir_of_highest_cumulative_flow_of_cell(&
        this,coords_in)
        class(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: this
        class(coords), allocatable :: coords_in
        class(coords), pointer :: highest_cumulative_flow_location
        class(*), pointer :: rdir
        type(generic_1d_coords) :: new_coarse_coords
            highest_cumulative_flow_location => &
                this%locate_highest_cumulative_flow_of_cell(coords_in)
            rdir => this%fine_rdirs_field%get_value(highest_cumulative_flow_location)
            select type (rdir)
            type is (integer)
                new_coarse_coords = &
                    this%iic_llp_dir_based_rdirs_calculate_new_coarse_coords(rdir, &
                                                       highest_cumulative_flow_location)
                call this%coarse_rdirs_field%set_value(coords_in,new_coarse_coords%index)
            end select
            deallocate(rdir)
    end subroutine irregular_dbr_assign_rdir_of_highest_cumulative_flow_of_cell

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

    !This function is just a dummy
    function irregular_latlon_dir_based_rdirs_generate_permitted_rdir(this,coords_in,&
                                                                      section_coords_in) &
                                                                  result(permitted_rdir)
        class(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: this
        class(coords) :: coords_in
        class(section_coords) :: section_coords_in
        class(direction_indicator), pointer :: permitted_rdir
            select type (coords_in)
            class is (coords)
                continue
            end select
            select type (section_coords_in)
            class is (section_coords)
                continue
            end select
            allocate(permitted_rdir,source=dir_based_direction_indicator(this%no_data_rdir))
    end function irregular_latlon_dir_based_rdirs_generate_permitted_rdir

    function dir_based_rdirs_get_no_data_rdir_value(this) result(no_data_rdir_value)
        class(latlon_dir_based_rdirs_loop_breaker) :: this
        class(direction_indicator), pointer :: no_data_rdir_value
            allocate(no_data_rdir_value,source=dir_based_direction_indicator(this%no_data_rdir))
    end function dir_based_rdirs_get_no_data_rdir_value

!   Repeat code due to restrictions of Fortan
    function irregular_dir_based_rdirs_get_no_data_rdir_value(this) result(no_data_rdir_value)
        class(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: this
        class(direction_indicator), pointer :: no_data_rdir_value
            allocate(no_data_rdir_value,source=dir_based_direction_indicator(this%no_data_rdir))
    end function irregular_dir_based_rdirs_get_no_data_rdir_value

    function iic_llp_dir_based_rdirs_calculate_new_coarse_coords(this,rdir_in, &
                                                                 fine_coords_in) &
            result(new_coarse_coords)
        class(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: this
        integer :: rdir_in
        class(coords) :: fine_coords_in
        type(latlon_coords) :: new_fine_coords
        type(generic_1d_coords) :: new_coarse_coords
        integer :: nlat,nlon
        integer :: new_coarse_cell_index
            select type(fine_rdirs => this%fine_rdirs_field)
            type is (latlon_field_section)
                nlat = fine_rdirs%get_nlat()
                nlon = fine_rdirs%get_nlon()
            end select
            select type(fine_coords_in)
            type is (latlon_coords)
                new_fine_coords = fine_coords_in
            end select
            if ( rdir_in == 7 .or. rdir_in == 8 .or. rdir_in == 9) then
                new_fine_coords%lat = new_fine_coords%lat - 1
            else if ( rdir_in == 1 .or. rdir_in == 2 .or. rdir_in == 3) then
                new_fine_coords%lat = new_fine_coords%lat + 1
            end if
            if ( rdir_in == 7 .or. rdir_in == 4 .or. rdir_in == 1 ) then
                new_fine_coords%lon = new_fine_coords%lon - 1
            else if ( rdir_in == 9 .or. rdir_in == 6 .or. rdir_in == 3 ) then
                new_fine_coords%lon = new_fine_coords%lon + 1
            end if
            if ( rdir_in == 5 .or. rdir_in == 0 .or. rdir_in == -1 ) then
                write(*,*) "Error in loop breaking code"
                stop
            end if
            if ( new_fine_coords%lat <= 0 ) then
                new_fine_coords%lat = nlat + new_fine_coords%lat
            end if
            if ( new_fine_coords%lon <= 0 ) then
                new_fine_coords%lon = nlon + new_fine_coords%lon
            end if
            if ( new_fine_coords%lat > nlat ) then
                new_fine_coords%lat =  new_fine_coords%lat - nlat
            end if
            if ( new_fine_coords%lon > nlon ) then
                new_fine_coords%lon = new_fine_coords%lon - nlon
            end if
            new_coarse_cell_index = this%expanded_cell_numbers_data(new_fine_coords%lat,&
                                                                    new_fine_coords%lon)
            new_coarse_coords = generic_1d_coords(new_coarse_cell_index)
    end function iic_llp_dir_based_rdirs_calculate_new_coarse_coords

    subroutine latlon_dir_based_rdirs_find_next_cell_downstream(this,coords_inout,&
                                                                outflow_cell_reached)
        class(latlon_dir_based_rdirs_loop_breaker) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: outflow_cell_reached
        class(*), pointer :: rdir
        select type (coords_inout)
        type is (latlon_coords)
            rdir => this%coarse_rdirs_field%get_value(coords_inout)
            select type (rdir)
            type is (integer)
                if ( rdir == 7 .or. rdir == 8 .or. rdir == 9) then
                    coords_inout%lat = coords_inout%lat - 1
                else if ( rdir == 1 .or. rdir == 2 .or. rdir == 3) then
                    coords_inout%lat = coords_inout%lat + 1
                end if
                if ( rdir == 7 .or. rdir == 4 .or. rdir == 1 ) then
                    coords_inout%lon = coords_inout%lon - 1
                else if ( rdir == 9 .or. rdir == 6 .or. rdir == 3 ) then
                    coords_inout%lon = coords_inout%lon + 1
                end if
                if ( rdir == 5 .or. rdir == 0 .or. rdir == -1 ) then
                    outflow_cell_reached =  .true.
                else
                    outflow_cell_reached =  .false.
                end if
            end select
            deallocate(rdir)
        end select
    end subroutine latlon_dir_based_rdirs_find_next_cell_downstream

    subroutine icon_icosohedral_cell_find_next_cell_downstream(this,coords_inout,&
                                                               outflow_cell_reached)
        class(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: outflow_cell_reached
        class(*), pointer :: next_cell_index
        select type (coords_inout)
        type is (generic_1d_coords)
            next_cell_index => this%fine_rdirs_field%get_value(coords_inout)
            select type (next_cell_index)
            type is (integer)
                if ( next_cell_index == this%index_for_sink .or. &
                     next_cell_index == this%index_for_outflow .or. &
                     next_cell_index == this%index_for_ocean_point ) then
                    outflow_cell_reached =  .false.
                else
                    outflow_cell_reached =  .false.
                    coords_inout%index = next_cell_index
                end if
            end select
            deallocate(next_cell_index)
        end select
    end subroutine icon_icosohedral_cell_find_next_cell_downstream

end module loop_breaker_mod
