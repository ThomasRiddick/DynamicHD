module area_mod
use coords_mod
use doubly_linked_list_mod
use subfield_mod
use field_section_mod
use precision_mod
use cotat_parameters_mod, only: MUFP, area_threshold, run_check_for_sinks
implicit none
private

type, abstract :: area
    private
        class(field_section), pointer :: river_directions => null()
        class(field_section), pointer :: total_cumulative_flow => null()
        !The following variables are only used by latlon based area's
        !(Have to insert here due to lack of multiple inheritance)
        integer :: section_min_lat
        integer :: section_min_lon
        integer :: section_max_lat
        integer :: section_max_lon
    contains
        private
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor
        procedure(destructor), deferred, public :: destructor
        procedure :: find_next_pixel_upstream
        procedure(find_upstream_neighbors), deferred, nopass :: find_upstream_neighbors
        procedure(check_if_coords_are_in_area), deferred, nopass :: check_if_coords_are_in_area
        procedure(neighbor_flows_to_pixel), deferred, nopass :: neighbor_flows_to_pixel
        procedure(find_next_pixel_downstream), deferred, nopass :: find_next_pixel_downstream
        procedure(print_area), deferred, nopass :: print_area
end type area

abstract interface
    !Must explicit pass the object this to this function as pointers to it have the nopass
    !attribute
    pure function check_if_coords_are_in_area(this,coords_in) result(within_area)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: within_area
    end function check_if_coords_are_in_area

    !Must explicit pass the object this to this function as pointers to it have the nopass
    !attribute
    function find_upstream_neighbors(this,coords_in) result(list_of_neighbors)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(coords), pointer :: list_of_neighbors(:)
    end function find_upstream_neighbors

    !Must explicit pass the object this to this function as pointers to it have the nopass
    !attribute
    function neighbor_flows_to_pixel(this,coords_in,flipped_direction_from_neighbor)
        import area
        import coords
        import direction_indicator
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        logical :: neighbor_flows_to_pixel
    end function neighbor_flows_to_pixel

    !Must explicitly pass the object this to this function as pointers to it have the nopass
    !attribute
    subroutine find_next_pixel_downstream(this,coords_inout,coords_not_in_area,outflow_pixel_reached)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
    end subroutine find_next_pixel_downstream

    !Must explicitly pass the object this to this function as pointers to it have the nopass
    !attribute
    subroutine print_area(this)
        import area
        implicit none
        class(area), intent(in) :: this
    end subroutine print_area

    subroutine destructor(this)
        import area
        class(area), intent(inout) :: this
    end subroutine destructor

end interface

type, extends(area), abstract :: field
    private
    !Despite the name this field section actually covers the entire field
    class(field_section), pointer :: cells_to_reprocess => null()
    class(section_coords), allocatable :: field_section_coords
    contains
        private
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor => field_destructor
        procedure, public :: destructor => field_destructor
        procedure :: check_cell_for_localized_loops
        procedure(check_field_for_localized_loops), deferred, public :: check_field_for_localized_loops
        procedure(initialize_cells_to_reprocess_field_section), deferred :: &
            initialize_cells_to_reprocess_field_section
end type field

abstract interface
    function check_field_for_localized_loops(this) result(cells_to_reprocess)
        import field
        implicit none
        class(field) :: this
        logical, dimension(:,:), pointer :: cells_to_reprocess
    end function check_field_for_localized_loops

    subroutine initialize_cells_to_reprocess_field_section(this)
        import field
        implicit none
        class(field) :: this
    end subroutine initialize_cells_to_reprocess_field_section
end interface

type, extends(area), abstract :: neighborhood
    private
    integer :: center_cell_id
    contains
        private
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor => neighborhood_destructor
        procedure, public :: destructor => neighborhood_destructor
        procedure :: find_downstream_cell
        procedure :: path_reenters_region
        procedure :: is_center_cell
        procedure(calculate_direction_indicator), deferred :: calculate_direction_indicator
        procedure(in_which_cell), deferred :: in_which_cell
end type neighborhood

abstract interface

    pure function in_which_cell(this,pixel) result(in_cell)
        import neighborhood
        import coords
        implicit none
        class(neighborhood), intent(in) :: this
        class(coords), intent(in) :: pixel
        integer :: in_cell
    end function in_which_cell

    pure function calculate_direction_indicator(this,downstream_cell) result(flow_direction)
        import neighborhood
        import direction_indicator
        implicit none
        class(neighborhood), intent(in) :: this
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
    end function calculate_direction_indicator
end interface

type, extends(area), abstract :: cell
    private
        class(subfield), pointer :: rejected_pixels => null()
        class(subfield), pointer :: cell_cumulative_flow => null()
        class(neighborhood), allocatable :: cell_neighborhood
        integer :: number_of_edge_pixels
        logical :: contains_river_mouths
        logical :: ocean_cell
        class(section_coords), allocatable :: cell_section_coords
    contains
    private
        procedure, public :: process_cell
        procedure, public :: set_contains_river_mouths
        procedure, public :: print_cell_and_neighborhood
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor => cell_destructor
        procedure, public :: destructor => cell_destructor
        procedure :: find_pixel_with_LUDA
        procedure :: find_pixel_with_LCDA
        procedure :: measure_upstream_path
        procedure :: check_MUFP
        procedure, public :: test_generate_cell_cumulative_flow
        procedure :: generate_cell_cumulative_flow
        procedure(find_edge_pixels), deferred :: find_edge_pixels
        procedure(calculate_length_through_pixel), deferred :: calculate_length_through_pixel
        procedure(initialize_cell_cumulative_flow_subfield), deferred :: &
            initialize_cell_cumulative_flow_subfield
        procedure(initialize_rejected_pixels_subfield), deferred :: &
            initialize_rejected_pixels_subfield
        procedure(generate_list_of_pixels_in_cell), deferred :: generate_list_of_pixels_in_cell
        procedure(get_flow_direction_for_sink), deferred :: get_flow_direction_for_sink
        procedure(get_flow_direction_for_outflow), deferred :: get_flow_direction_for_outflow
        procedure(get_flow_direction_for_ocean_point), deferred :: get_flow_direction_for_ocean_point
        procedure(get_sink_combined_cumulative_flow), deferred :: get_sink_combined_cumulative_flow
        procedure(get_rmouth_outflow_combined_cumulative_flow), deferred :: &
            get_rmouth_outflow_combined_cumulative_flow
        procedure(mark_ocean_and_river_mouth_points), deferred, public :: &
            mark_ocean_and_river_mouth_points
end type cell

abstract interface
    pure function find_edge_pixels(this) result(edge_pixel_coords_list)
        import cell
        import coords
        implicit none
        class(cell), intent(in) :: this
        class(coords), pointer :: edge_pixel_coords_list(:)
    end function find_edge_pixels

    function calculate_length_through_pixel(this,coords_in) result(length_through_pixel)
        import cell
        import coords
        import double_precision
        implicit none
        class(cell), intent(in) :: this
        class(coords), intent(in) :: coords_in
        real(kind=double_precision) :: length_through_pixel
    end function calculate_length_through_pixel

    subroutine initialize_cell_cumulative_flow_subfield(this)
        import cell
        implicit none
        class(cell), intent(inout) :: this
    end subroutine initialize_cell_cumulative_flow_subfield

    subroutine initialize_rejected_pixels_subfield(this)
        import cell
        implicit none
        class(cell), intent(inout) :: this
    end subroutine initialize_rejected_pixels_subfield

    function generate_list_of_pixels_in_cell(this) result(list_of_pixels_in_cell)
        import cell
        import doubly_linked_list
        implicit none
        class(cell), intent(in) :: this
        type(doubly_linked_list), pointer :: list_of_pixels_in_cell
    end function generate_list_of_pixels_in_cell

    pure function get_flow_direction_for_sink(this) result(flow_direction_for_sink)
        import cell
        import coords
        import direction_indicator
        implicit none
        class(cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_sink
    end function get_flow_direction_for_sink

    pure function get_flow_direction_for_outflow(this) result(flow_direction_for_outflow)
        import cell
        import coords
        import direction_indicator
        implicit none
        class(cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_outflow
    end function get_flow_direction_for_outflow

    pure function get_flow_direction_for_ocean_point(this) result(flow_direction_for_ocean_point)
        import cell
        import coords
        import direction_indicator
        implicit none
        class(cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
    end function get_flow_direction_for_ocean_point

    function get_sink_combined_cumulative_flow(this) &
        result(combined_cumulative_flow_to_sinks)
        import cell
        class(cell), intent(in) :: this
        integer :: combined_cumulative_flow_to_sinks
    end function get_sink_combined_cumulative_flow

    function get_rmouth_outflow_combined_cumulative_flow(this) &
        result(combined_cumulative_rmouth_outflow)
        import cell
        class(cell), intent(in) :: this
        integer :: combined_cumulative_rmouth_outflow
    end function get_rmouth_outflow_combined_cumulative_flow

    subroutine mark_ocean_and_river_mouth_points(this)
        import cell
        class(cell), intent(inout) :: this
    end subroutine
end interface

type, extends(field), abstract :: latlon_field
    integer :: section_width_lat
    integer :: section_width_lon
    contains
        private
        procedure, public :: init_latlon_field
        procedure, public :: check_field_for_localized_loops => latlon_check_field_for_localized_loops
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        procedure, nopass :: print_area => latlon_print_area
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        procedure :: initialize_cells_to_reprocess_field_section => &
            latlon_initialize_cells_to_reprocess_field_section
end type latlon_field

type, extends(neighborhood), abstract :: latlon_neighborhood
    private
        integer :: center_cell_min_lat
        integer :: center_cell_min_lon
        integer :: center_cell_max_lat
        integer :: center_cell_max_lon
    contains
        procedure :: init_latlon_neighborhood
        procedure :: in_which_cell => latlon_in_which_cell
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        procedure, nopass :: print_area => latlon_print_area
end type latlon_neighborhood

type, extends(cell), abstract :: latlon_cell

    integer :: section_width_lat
    integer :: section_width_lon
    contains
        procedure :: init_latlon_cell
        procedure :: find_edge_pixels => latlon_find_edge_pixels
        procedure :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        procedure :: initialize_cell_cumulative_flow_subfield => &
            latlon_initialize_cell_cumulative_flow_subfield
        procedure :: initialize_rejected_pixels_subfield => latlon_initialize_rejected_pixels_subfield
        procedure :: generate_list_of_pixels_in_cell => latlon_generate_list_of_pixels_in_cell
        procedure :: get_sink_combined_cumulative_flow => latlon_get_sink_combined_cumulative_flow
        procedure :: get_rmouth_outflow_combined_cumulative_flow => &
            latlon_get_rmouth_outflow_combined_cumulative_flow
        procedure, nopass :: print_area => latlon_print_area
        procedure(is_diagonal), deferred :: is_diagonal
end type latlon_cell

abstract interface
    function is_diagonal(this,coords_in) result(diagonal)
        import latlon_cell
        import coords
        implicit none
        class(latlon_cell), intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: diagonal
    end function is_diagonal
end interface

type, public, extends(latlon_field) :: latlon_dir_based_rdirs_field
    private
    contains
        private
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
end type latlon_dir_based_rdirs_field

interface  latlon_dir_based_rdirs_field
    procedure latlon_dir_based_rdirs_field_constructor
end interface

type, extends(latlon_neighborhood) :: latlon_dir_based_rdirs_neighborhood
    private
    contains
    private
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        procedure :: calculate_direction_indicator => dir_based_rdirs_calculate_direction_indicator
end type latlon_dir_based_rdirs_neighborhood

interface latlon_dir_based_rdirs_neighborhood
    procedure latlon_dir_based_rdirs_neighborhood_constructor
end interface latlon_dir_based_rdirs_neighborhood

type, public, extends(latlon_cell) :: latlon_dir_based_rdirs_cell
    integer :: flow_direction_for_sink = 5
    integer :: flow_direction_for_outflow = 0
    integer :: flow_direction_for_ocean_point = -1
contains
    procedure :: is_diagonal => dir_based_rdirs_is_diagonal
    procedure, nopass :: neighbor_flows_to_pixel => &
        latlon_dir_based_rdirs_neighbor_flows_to_pixel
    procedure :: init_dir_based_rdirs_latlon_cell
    procedure :: get_flow_direction_for_sink => latlon_dir_based_rdirs_get_flow_direction_for_sink
    procedure :: get_flow_direction_for_outflow => latlon_dir_based_rdirs_get_flow_direction_for_outflow
    procedure :: get_flow_direction_for_ocean_point => latlon_dir_based_rdirs_get_flow_direction_for_ocean_point
    procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
    procedure, public :: mark_ocean_and_river_mouth_points => latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points
end type latlon_dir_based_rdirs_cell

interface latlon_dir_based_rdirs_cell
    procedure latlon_dir_based_rdirs_cell_constructor
end interface latlon_dir_based_rdirs_cell

contains

    subroutine find_next_pixel_upstream(this,coords_inout,coords_not_in_area)
        class(area), intent(in) :: this
        class(coords), allocatable, intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        class(coords), pointer :: upstream_neighbors_coords_list(:)
        class(*), pointer :: upstream_neighbor_total_cumulative_flow
        integer :: max_cmltv_flow_working_value
        integer :: i
        max_cmltv_flow_working_value = 0
        upstream_neighbors_coords_list => this%find_upstream_neighbors(this,coords_inout)
        if (size(upstream_neighbors_coords_list) > 0) then
            do i = 1, size(upstream_neighbors_coords_list)
                upstream_neighbor_total_cumulative_flow => &
                    this%total_cumulative_flow%get_value(upstream_neighbors_coords_list(i))
                select type (upstream_neighbor_total_cumulative_flow)
                type is (integer)
                    if ( upstream_neighbor_total_cumulative_flow > max_cmltv_flow_working_value) then
                        if (allocated(coords_inout)) deallocate(coords_inout)
                        allocate(coords_inout,source=upstream_neighbors_coords_list(i))
                        max_cmltv_flow_working_value = upstream_neighbor_total_cumulative_flow
                    end if
                end select
                deallocate(upstream_neighbor_total_cumulative_flow)
            end do
            coords_not_in_area = .not. this%check_if_coords_are_in_area(this,coords_inout)
        else
            coords_not_in_area = .TRUE.
        end if
        deallocate(upstream_neighbors_coords_list)
    end subroutine find_next_pixel_upstream

    subroutine field_destructor(this)
        class(field), intent(inout) :: this
            if (associated(this%cells_to_reprocess)) call this%cells_to_reprocess%deallocate_data()
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (associated(this%cells_to_reprocess)) deallocate(this%cells_to_reprocess)
    end subroutine field_destructor

    function check_cell_for_localized_loops(this,coords_inout) result(localized_loop_found)
        class(field)  :: this
        class(coords) :: coords_inout
        logical :: localized_loop_found
        class(coords),allocatable :: initial_cell_coords
        logical :: coords_not_in_area_dummy
        logical :: is_sea_point_or_horizontal_edge_or_sink
            allocate(initial_cell_coords,source=coords_inout)
            call this%find_next_pixel_downstream(this,coords_inout,coords_not_in_area_dummy)
            if (coords_inout%are_equal_to(initial_cell_coords)) then
                is_sea_point_or_horizontal_edge_or_sink = .True.
            else
                is_sea_point_or_horizontal_edge_or_sink = .False.
            end if
            call this%find_next_pixel_downstream(this,coords_inout,coords_not_in_area_dummy)
            if (coords_inout%are_equal_to(initial_cell_coords) .and. &
                .not. is_sea_point_or_horizontal_edge_or_sink) then
                localized_loop_found = .True.
            else
                localized_loop_found = .False.
            end if
            deallocate(initial_cell_coords)
    end function check_cell_for_localized_loops

    function process_cell(this) result(flow_direction)
        class(cell) :: this
        class(*), allocatable :: downstream_cell
        class(coords), pointer :: LCDA_pixel_coords => null()
        class(coords), pointer :: LUDA_pixel_coords => null()
        class(direction_indicator), pointer :: flow_direction
        class(*), pointer :: outlet_pixel_cumulative_flow_value
        logical :: no_remaining_outlets
        integer :: outlet_pixel_cumulative_flow
        integer :: cumulative_sink_outflow
        integer :: cumulative_rmouth_outflow
            if (.not. this%ocean_cell) then
                call this%generate_cell_cumulative_flow()
                LCDA_pixel_coords=>this%find_pixel_with_LCDA()
                do
                    if (associated(LUDA_pixel_coords)) deallocate(LUDA_pixel_coords)
                    LUDA_pixel_coords=>this%find_pixel_with_LUDA(no_remaining_outlets)
                    if (no_remaining_outlets) then
                        exit
                    else
                        if (this%check_MUFP(LUDA_pixel_coords,LCDA_pixel_coords)) exit
                    end if
                end do
                deallocate(LCDA_pixel_coords)
                if (no_remaining_outlets) then
                    outlet_pixel_cumulative_flow = -1
                else
                    outlet_pixel_cumulative_flow_value => &
                        this%total_cumulative_flow%get_value(LUDA_pixel_coords)
                    select type(outlet_pixel_cumulative_flow_value)
                    type is (integer)
                        outlet_pixel_cumulative_flow = outlet_pixel_cumulative_flow_value
                    end select
                    deallocate(outlet_pixel_cumulative_flow_value)
                end if
                if (run_check_for_sinks) then
                    cumulative_sink_outflow = this%get_sink_combined_cumulative_flow()
                else
                    cumulative_sink_outflow = 0
                end if
                if (this%contains_river_mouths) then
                    cumulative_rmouth_outflow = this%get_rmouth_outflow_combined_cumulative_flow()
                else
                    cumulative_rmouth_outflow = -1
                end if
                if ( cumulative_sink_outflow > cumulative_rmouth_outflow .and. &
                     cumulative_sink_outflow > outlet_pixel_cumulative_flow ) then
                    flow_direction => this%get_flow_direction_for_sink()
                    deallocate(LUDA_pixel_coords)
                    return
                else if (  cumulative_rmouth_outflow > outlet_pixel_cumulative_flow) then
                    flow_direction => this%get_flow_direction_for_outflow()
                    deallocate(LUDA_pixel_coords)
                    return
                end if
                allocate(downstream_cell,source=this%cell_neighborhood%find_downstream_cell(LUDA_pixel_coords))
                select type (downstream_cell)
                type is (integer)
                    flow_direction => this%cell_neighborhood%calculate_direction_indicator(downstream_cell)
                end select
                deallocate(LUDA_pixel_coords)
            else
                flow_direction => this%get_flow_direction_for_ocean_point()
            end if
    end function process_cell

    subroutine print_cell_and_neighborhood(this)
    class(cell), intent(in) :: this
        write(*,*) 'For cell'
        call this%print_area(this)
        write (*,*) 'contains_river_mouths= ', this%contains_river_mouths
        write (*,*) 'ocean_cell= ', this%ocean_cell
        write(*,*) 'For neighborhood'
        call this%cell_neighborhood%print_area(this%cell_neighborhood)
    end subroutine

    subroutine cell_destructor(this)
        class(cell), intent(inout) :: this
            if (associated(this%rejected_pixels)) then
                call this%rejected_pixels%destructor()
                deallocate(this%rejected_pixels)
            end if
            if (associated(this%cell_cumulative_flow)) then
                call this%cell_cumulative_flow%destructor()
                deallocate(this%cell_cumulative_flow)
            end if
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (allocated(this%cell_neighborhood)) call this%cell_neighborhood%destructor()
    end subroutine cell_destructor

    function find_pixel_with_LUDA(this,no_remaining_outlets) result(LUDA_pixel_coords)
        class(cell), intent(in) :: this
        class(coords), pointer :: LUDA_pixel_coords
        logical, intent(out) :: no_remaining_outlets
        class(coords), allocatable :: working_pixel_coords
        integer :: LUDA_working_value
        integer :: i
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(*), pointer :: rejected_pixel
        class(*), pointer :: upstream_drainage
        logical :: coords_not_in_cell
        nullify(LUDA_pixel_coords)
        edge_pixel_coords_list => this%find_edge_pixels()
            LUDA_working_value = 0
            do i = 1,size(edge_pixel_coords_list)
                rejected_pixel => this%rejected_pixels%get_value(edge_pixel_coords_list(i))
                select type(rejected_pixel)
                type is (logical)
                    if ( .NOT. rejected_pixel) then
                        upstream_drainage => this%total_cumulative_flow%get_value(edge_pixel_coords_list(i))
                        select type (upstream_drainage)
                        type is (integer)
                        if (upstream_drainage > LUDA_working_value) then
                                allocate(working_pixel_coords,source=edge_pixel_coords_list(i))
                                call this%find_next_pixel_downstream(this,working_pixel_coords,coords_not_in_cell)
                                deallocate(working_pixel_coords)
                                if(coords_not_in_cell) then
                                    LUDA_working_value = upstream_drainage
                                    if (associated(LUDA_pixel_coords)) deallocate(LUDA_pixel_coords)
                                    allocate(LUDA_pixel_coords,source=edge_pixel_coords_list(i))
                                end if
                            end if
                        end select
                        deallocate(upstream_drainage)
                    end if
                end select
                deallocate(rejected_pixel)
            end do
            if (.not. associated(LUDA_pixel_coords)) then
                allocate(LUDA_pixel_coords,source=edge_pixel_coords_list(1))
                no_remaining_outlets=.True.
            else
                no_remaining_outlets=.False.
            end if
            deallocate(edge_pixel_coords_list)
    end function find_pixel_with_LUDA

    function find_pixel_with_LCDA(this) result(LCDA_pixel_coords)
        class(cell), intent(in) :: this
        class(coords), pointer :: LCDA_pixel_coords
        class(coords), allocatable :: working_pixel_coords
        integer :: LCDA_pixel_working_value
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(*), pointer :: cell_drainage_area
        logical :: coords_not_in_cell
        integer :: i
            nullify(LCDA_pixel_coords)
            edge_pixel_coords_list => this%find_edge_pixels()
            LCDA_pixel_working_value = 0
            do i = 1,size(edge_pixel_coords_list)
                cell_drainage_area => this%cell_cumulative_flow%get_value(edge_pixel_coords_list(i))
                select type(cell_drainage_area)
                type is (integer)
                    if (cell_drainage_area > LCDA_pixel_working_value) then
                        allocate(working_pixel_coords,source=edge_pixel_coords_list(i))
                        call this%find_next_pixel_downstream(this,working_pixel_coords,coords_not_in_cell)
                        deallocate(working_pixel_coords)
                        if(coords_not_in_cell) then
                            LCDA_pixel_working_value = cell_drainage_area
                            if (associated(LCDA_pixel_coords)) deallocate(LCDA_pixel_coords)
                            allocate(LCDA_pixel_coords,source=edge_pixel_coords_list(i))
                        end if
                    end if
                end select
                deallocate(cell_drainage_area)
            end do
            if (.not. associated(LCDA_pixel_coords)) then
                allocate(LCDA_pixel_coords,source=edge_pixel_coords_list(1))
            end if
            deallocate(edge_pixel_coords_list)
    end function find_pixel_with_LCDA

    function measure_upstream_path(this,outlet_pixel_coords) result(upstream_path_length)
        class(cell), intent(in) :: this
        class(coords), intent(in), target :: outlet_pixel_coords
        class(coords), allocatable :: working_coords
        logical :: coords_not_in_cell
        real(kind=double_precision) :: length_through_pixel
        real(kind=double_precision) :: upstream_path_length
            upstream_path_length = 0.0_double_precision
            allocate(working_coords, source=outlet_pixel_coords)
            do
                length_through_pixel = this%calculate_length_through_pixel(working_coords)
                do
                    upstream_path_length = upstream_path_length + length_through_pixel
                    call this%find_next_pixel_upstream(working_coords,coords_not_in_cell)
                    length_through_pixel = this%calculate_length_through_pixel(working_coords)
                    if (coords_not_in_cell) exit
                end do
                if (.NOT. this%cell_neighborhood%path_reenters_region(working_coords)) exit
            end do
            deallocate(working_coords)
    end function measure_upstream_path

    function check_MUFP(this,LUDA_pixel_coords,LCDA_pixel_coords) result(accept_pixel)
        class(cell) :: this
        class(coords), intent(in) :: LUDA_pixel_coords
        class(coords), intent(in) :: LCDA_pixel_coords
        logical :: accept_pixel
            if (this%measure_upstream_path(LUDA_pixel_coords) > MUFP) then
                accept_pixel = .TRUE.
            else if (LUDA_pixel_coords%are_equal_to(LCDA_pixel_coords)) then
                accept_pixel = .TRUE.
            else
                call this%rejected_pixels%set_value(LUDA_pixel_coords,.TRUE.)
                accept_pixel = .FALSE.
            end if
    end function check_MUFP

    function test_generate_cell_cumulative_flow(this) result(cmltv_flow_array)
        class(cell), intent(inout) :: this
        class(*), dimension(:,:), pointer :: cmltv_flow_array
        call this%generate_cell_cumulative_flow()
        select type(cmltv_flow => this%cell_cumulative_flow)
        type is (latlon_subfield)
            cmltv_flow_array => cmltv_flow%latlon_get_data()
        end select
    end function test_generate_cell_cumulative_flow

    subroutine generate_cell_cumulative_flow(this)
        class(cell), intent(inout) :: this
        class(doubly_linked_list), pointer :: pixels_to_process
        class(coords), pointer :: upstream_neighbors_coords_list(:)
        class(*), pointer :: pixel_coords
        class(*), allocatable :: neighbor_coords
        logical :: unprocessed_inflowing_pixels
        integer :: pixel_cumulative_flow
        class(*), pointer :: neighbor_cumulative_flow
        integer :: i
            call this%initialize_cell_cumulative_flow_subfield()
            pixels_to_process => this%generate_list_of_pixels_in_cell()
            do
                if (pixels_to_process%get_length() <= 0) exit
                do
                    !Combine iteration and check to see if we have reached the end
                    !of the list
                    if(pixels_to_process%iterate_forward()) exit
                    pixel_cumulative_flow = 1
                    unprocessed_inflowing_pixels = .FALSE.
                    pixel_coords => pixels_to_process%get_value_at_iterator_position()
                    select type (pixel_coords)
                    class is (coords)
                        upstream_neighbors_coords_list => this%find_upstream_neighbors(this,pixel_coords)
                    end select
                    if (size(upstream_neighbors_coords_list) > 0) then
                        do i = 1,size(upstream_neighbors_coords_list)
                            if (allocated(neighbor_coords)) deallocate(neighbor_coords)
                            allocate(neighbor_coords,source=upstream_neighbors_coords_list(i))
                            select type (neighbor_coords)
                            class is (coords)
                                if( .not. this%check_if_coords_are_in_area(this,neighbor_coords)) then
                                    cycle
                                end if
                                neighbor_cumulative_flow => this%cell_cumulative_flow%get_value(neighbor_coords)
                            end select
                            select type (neighbor_cumulative_flow)
                            type is (integer)
                                if (neighbor_cumulative_flow == 0) then
                                    unprocessed_inflowing_pixels = .TRUE.
                                else
                                    pixel_cumulative_flow = pixel_cumulative_flow + neighbor_cumulative_flow
                                end if
                            end select
                            deallocate(neighbor_cumulative_flow)
                            if (unprocessed_inflowing_pixels) exit
                        end do
                        if (allocated(neighbor_coords)) deallocate(neighbor_coords)
                    end if
                    if (.not. unprocessed_inflowing_pixels) then
                        select type (pixel_coords)
                        class is (coords)
                            call this%cell_cumulative_flow%set_value(pixel_coords,pixel_cumulative_flow)
                        end select
                        call pixels_to_process%remove_element_at_iterator_position()
                        if (pixels_to_process%get_length() <= 0) exit
                    end if
                    deallocate(upstream_neighbors_coords_list)
                end do
                if (associated(upstream_neighbors_coords_list)) deallocate(upstream_neighbors_coords_list)
                call pixels_to_process%reset_iterator()
            end do
            deallocate(pixels_to_process)
    end subroutine generate_cell_cumulative_flow

    subroutine set_contains_river_mouths(this,value)
        class(cell) :: this
        logical :: value
            this%contains_river_mouths = value
    end subroutine

    subroutine init_latlon_field(this,field_section_coords,river_directions)
        class(latlon_field) :: this
        class(*), dimension(:,:), pointer :: river_directions
        type(latlon_section_coords) :: field_section_coords
            this%section_min_lat = field_section_coords%section_min_lat
            this%section_min_lon = field_section_coords%section_min_lon
            this%section_width_lat = field_section_coords%section_width_lat
            this%section_width_lon = field_section_coords%section_width_lon
            this%section_max_lat =  field_section_coords%section_min_lat + field_section_coords%section_width_lat - 1
            this%section_max_lon =  field_section_coords%section_min_lon + field_section_coords%section_width_lon - 1
            allocate(this%field_section_coords,source=field_section_coords)
            this%total_cumulative_flow => null()
            this%river_directions => latlon_field_section(river_directions,field_section_coords)
            call this%initialize_cells_to_reprocess_field_section()
    end subroutine init_latlon_field

    function latlon_check_field_for_localized_loops(this) result(cells_to_reprocess)
        implicit none
        class(latlon_field) :: this
        class(*), dimension(:,:), pointer :: cells_to_reprocess_data
        logical, dimension(:,:), pointer :: cells_to_reprocess
        type(latlon_coords) :: cell_coords
        integer :: i,j
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    cell_coords = latlon_coords(i,j)
                    if (this%check_cell_for_localized_loops(cell_coords)) then
                        call this%cells_to_reprocess%set_value(cell_coords,.True.)
                    end if
                end do
            end do
            select type (cells_to_reprocess_field_section => this%cells_to_reprocess)
            type is (latlon_field_section)
                cells_to_reprocess_data => cells_to_reprocess_field_section%get_data()
                select type (cells_to_reprocess_data)
                type is (logical)
                    cells_to_reprocess => cells_to_reprocess_data
                end select
            end select
    end function latlon_check_field_for_localized_loops

    subroutine latlon_initialize_cells_to_reprocess_field_section(this)
        class(latlon_field) :: this
        class(*), dimension(:,:), pointer :: cells_to_reprocess_initialization_data
            allocate(logical:: cells_to_reprocess_initialization_data(this%section_width_lat,&
                                                                      this%section_width_lon))
            select type(cells_to_reprocess_initialization_data)
            type is (logical)
                cells_to_reprocess_initialization_data = .FALSE.
            end select
            select type (field_section_coords => this%field_section_coords)
            type is (latlon_section_coords)
                this%cells_to_reprocess => latlon_field_section(cells_to_reprocess_initialization_data, &
                                                                field_section_coords)
            end select
    end subroutine latlon_initialize_cells_to_reprocess_field_section

    subroutine init_latlon_cell(this,cell_section_coords,river_directions,&
                                total_cumulative_flow)
        class(latlon_cell), intent(inout) :: this
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
            allocate(this%cell_section_coords,source=cell_section_coords)
            this%section_min_lat = cell_section_coords%section_min_lat
            this%section_min_lon = cell_section_coords%section_min_lon
            this%section_width_lat = cell_section_coords%section_width_lat
            this%section_width_lon = cell_section_coords%section_width_lon
            this%section_max_lat =  cell_section_coords%section_min_lat + cell_section_coords%section_width_lat - 1
            this%section_max_lon =  cell_section_coords%section_min_lon + cell_section_coords%section_width_lon - 1
            river_directions_pointer => river_directions
            total_cumulative_flow_pointer => total_cumulative_flow
            this%river_directions => latlon_field_section(river_directions_pointer,cell_section_coords)
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,cell_section_coords)
            !Minus four to avoid double counting the corners
            this%number_of_edge_pixels = 2*this%section_width_lat + 2*this%section_width_lon - 4
            call this%initialize_rejected_pixels_subfield()
            this%contains_river_mouths = .FALSE.
            this%ocean_cell = .FALSE.
    end subroutine

    pure function latlon_find_edge_pixels(this) result(edge_pixel_coords_list)
        class(latlon_cell), intent(in) :: this
        class(coords), pointer :: edge_pixel_coords_list(:)
        integer :: i
        integer :: list_index
        list_index = 1
        allocate(latlon_coords::edge_pixel_coords_list(this%number_of_edge_pixels))
        select type (edge_pixel_coords_list)
        type is (latlon_coords)
            do i = this%section_min_lat,this%section_max_lat
                edge_pixel_coords_list(list_index) = latlon_coords(i,this%section_min_lon)
                edge_pixel_coords_list(list_index+1) = latlon_coords(i,this%section_max_lon)
                list_index = list_index + 2
            end do
            do  i = this%section_min_lon+1,this%section_max_lon-1
                edge_pixel_coords_list(list_index) = latlon_coords(this%section_min_lat,i)
                edge_pixel_coords_list(list_index+1) = latlon_coords(this%section_max_lat,i)
                list_index = list_index + 2
            end do
        end select
    end function latlon_find_edge_pixels

    function latlon_calculate_length_through_pixel(this,coords_in) result(length_through_pixel)
        class(latlon_cell), intent(in) :: this
        class(coords), intent(in) :: coords_in
        real(kind=double_precision) length_through_pixel
        if (this%is_diagonal(coords_in)) then
            length_through_pixel = 1.414_double_precision
        else
            length_through_pixel = 1.0_double_precision
        end if
    end function latlon_calculate_length_through_pixel

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    pure function latlon_check_if_coords_are_in_area(this,coords_in) result(within_area)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: within_area
        select type (coords_in)
        type is (latlon_coords)
            if (coords_in%lat >= this%section_min_lat .and. &
                coords_in%lat <= this%section_max_lat .and. &
                coords_in%lon >= this%section_min_lon .and. &
                coords_in%lon <= this%section_max_lon) then
                within_area = .TRUE.
            else
                within_area = .FALSE.
            end if
        end select
    end function latlon_check_if_coords_are_in_area

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_find_upstream_neighbors(this,coords_in) result(list_of_neighbors)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        type(latlon_coords), dimension(9) :: list_of_neighbors_temp
        class(coords), dimension(:), pointer :: list_of_neighbors
        integer :: i,j
        integer :: counter
        integer :: n
        counter = 0
        n = 0
        select type (coords_in)
        type is (latlon_coords)
            do i = coords_in%lat + 1, coords_in%lat - 1,-1
                do j = coords_in%lon - 1, coords_in%lon + 1
                    counter = counter + 1
                    if (counter == 5) cycle
                    if (this%neighbor_flows_to_pixel(this,latlon_coords(lat=i,lon=j),&
                        dir_based_direction_indicator(counter))) then
                        n = n + 1
                        list_of_neighbors_temp(n) = latlon_coords(lat=i,lon=j)
                    end if
                end do
            end do
        end select
        allocate(latlon_coords::list_of_neighbors(n))
        select type(list_of_neighbors)
        type is (latlon_coords)
            do counter = 1,n
                list_of_neighbors(counter) = list_of_neighbors_temp(counter)
            end do
        end select
    end function latlon_find_upstream_neighbors

    subroutine latlon_initialize_cell_cumulative_flow_subfield(this)
        class(latlon_cell), intent(inout) :: this
        class(*), dimension(:,:), pointer :: input_data
            allocate(integer::input_data(this%section_width_lat,this%section_width_lon))
            select type(input_data)
            type is (integer)
                input_data = 0
            end select
            select type(cell_section_coords => this%cell_section_coords)
            type is (latlon_section_coords)
                this%cell_cumulative_flow => latlon_subfield(input_data,cell_section_coords)
            end select
    end subroutine latlon_initialize_cell_cumulative_flow_subfield

    subroutine latlon_initialize_rejected_pixels_subfield(this)
        class(latlon_cell), intent(inout) :: this
        class(*), dimension(:,:), pointer :: rejected_pixel_initialization_data
            allocate(logical::rejected_pixel_initialization_data(this%section_width_lat,&
                                                                 this%section_width_lon))
            select type(rejected_pixel_initialization_data)
            type is (logical)
                rejected_pixel_initialization_data = .FALSE.
            end select
            select type (cell_section_coords => this%cell_section_coords)
            type is (latlon_section_coords)
                this%rejected_pixels => latlon_subfield(rejected_pixel_initialization_data,cell_section_coords)
            end select
    end subroutine latlon_initialize_rejected_pixels_subfield

    function latlon_generate_list_of_pixels_in_cell(this) result(list_of_pixels_in_cell)
        class(latlon_cell), intent(in) :: this
        type(doubly_linked_list), pointer :: list_of_pixels_in_cell
        integer :: i,j
            allocate(list_of_pixels_in_cell)
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    call list_of_pixels_in_cell%add_value_to_back(latlon_coords(i,j))
                end do
            end do
    end function

    function latlon_get_sink_combined_cumulative_flow(this) &
        result(combined_cumulative_flow_to_sinks)
        class(latlon_cell), intent(in) :: this
        integer :: combined_cumulative_flow_to_sinks
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_sink
        integer :: i,j
            combined_cumulative_flow_to_sinks = 0
            flow_direction_for_sink => this%get_flow_direction_for_sink()
            do j = this%section_min_lon, this%section_min_lon + this%section_width_lon - 1
                do i = this%section_min_lat, this%section_min_lat + this%section_width_lat - 1
                    current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                    current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                    select type (current_pixel_total_cumulative_flow)
                    type is (integer)
                        if (flow_direction_for_sink%is_equal_to(current_pixel_flow_direction)) then
                            combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow + &
                                combined_cumulative_flow_to_sinks
                        end if
                        deallocate(current_pixel_flow_direction)
                    end select
                    deallocate(current_pixel_total_cumulative_flow)
                end do
            end do
            deallocate(flow_direction_for_sink)
    end function latlon_get_sink_combined_cumulative_flow

    function latlon_get_rmouth_outflow_combined_cumulative_flow(this) &
        result(combined_cumulative_rmouth_outflow)
        class(latlon_cell), intent(in) :: this
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_outflow
        integer :: combined_cumulative_rmouth_outflow
        integer :: i,j
            combined_cumulative_rmouth_outflow = 0
            flow_direction_for_outflow => this%get_flow_direction_for_outflow()
            do j = this%section_min_lon, this%section_min_lon + this%section_width_lon - 1
                do i = this%section_min_lat, this%section_min_lat + this%section_width_lat - 1
                    current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                    current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                    select type (current_pixel_total_cumulative_flow)
                    type is (integer)
                        if (flow_direction_for_outflow%is_equal_to(current_pixel_flow_direction)) then
                            combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow + &
                                combined_cumulative_rmouth_outflow
                        end if
                    end select
                    deallocate(current_pixel_total_cumulative_flow)
                    deallocate(current_pixel_flow_direction)
                end do
            end do
            deallocate(flow_direction_for_outflow)
    end function latlon_get_rmouth_outflow_combined_cumulative_flow

    subroutine latlon_print_area(this)
    class(area), intent(in) :: this
    character(len=80) :: line
    class(*), pointer :: value
    integer :: i,j
        write(*,*) 'Area Information'
        write(*,*) 'min lat=', this%section_min_lat
        write(*,*) 'min lon=', this%section_min_lon
        write(*,*) 'max lat=', this%section_max_lat
        write(*,*) 'max lon=', this%section_max_lon
        write(*,*) 'area rdirs'
        select type(rdirs => this%river_directions)
        type is (latlon_field_section)
            do i = this%section_min_lat,this%section_max_lat
                line = ''
                do j = this%section_min_lon,this%section_max_lon
                    value => rdirs%get_value(latlon_coords(i,j))
                    select type(value)
                    type is (integer)
                        write(line,'(A,I5)') trim(line), value
                    end select
                    deallocate(value)
                end do
                write (*,'(A)') line
            end do
        end select
        write(*,*) 'area total cumulative flow'
        select type(total_cumulative_flow => this%total_cumulative_flow)
        type is (latlon_field_section)
            do i = this%section_min_lat,this%section_max_lat
                line=''
                do j = this%section_min_lon,this%section_max_lon
                    value => total_cumulative_flow%get_value(latlon_coords(i,j))
                    select type(value)
                    type is (integer)
                        write(line,'(A,I5)') trim(line), value
                    end select
                    deallocate(value)
                end do
                write (*,'(A)') line
            end do
        end select
    end subroutine latlon_print_area

    function latlon_dir_based_rdirs_cell_constructor(cell_section_coords,river_directions,&
                                                    total_cumulative_flow) result(constructor)
        type(latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            allocate(constructor)
            call constructor%init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,&
                                                              total_cumulative_flow)
    end function

    subroutine init_dir_based_rdirs_latlon_cell(this,cell_section_coords,river_directions,&
                                                total_cumulative_flow)
        class(latlon_dir_based_rdirs_cell) :: this
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            call this%init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
            allocate(this%cell_neighborhood, source=latlon_dir_based_rdirs_neighborhood(cell_section_coords, &
                river_directions, total_cumulative_flow))
    end subroutine init_dir_based_rdirs_latlon_cell

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_dir_based_rdirs_neighbor_flows_to_pixel(this,coords_in, &
                  flipped_direction_from_neighbor) result(neighbor_flows_to_pixel)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        class(*), pointer :: rdir
        logical neighbor_flows_to_pixel
        integer, dimension(9) :: directions_pointing_to_pixel
        directions_pointing_to_pixel = (/9,8,7,6,5,4,3,2,1/)
        select type(flipped_direction_from_neighbor)
        type is (dir_based_direction_indicator)
            rdir => this%river_directions%get_value(coords_in)
            select type(rdir)
            type is (integer)
                if ( rdir == &
                    directions_pointing_to_pixel(flipped_direction_from_neighbor%get_direction())) then
                    neighbor_flows_to_pixel = .TRUE.
                else
                    neighbor_flows_to_pixel = .FALSE.
                end if
            end select
            deallocate(rdir)
        end select
    end function latlon_dir_based_rdirs_neighbor_flows_to_pixel

    pure function latlon_dir_based_rdirs_get_flow_direction_for_sink(this) &
        result(flow_direction_for_sink)
        class(latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_sink
            allocate(flow_direction_for_sink,source=&
                dir_based_direction_indicator(this%flow_direction_for_sink))
    end function latlon_dir_based_rdirs_get_flow_direction_for_sink

    pure function latlon_dir_based_rdirs_get_flow_direction_for_outflow(this) &
        result(flow_direction_for_outflow)
        class(latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_outflow
            allocate(flow_direction_for_outflow,source=&
                dir_based_direction_indicator(this%flow_direction_for_outflow))
    end function latlon_dir_based_rdirs_get_flow_direction_for_outflow

    pure function latlon_dir_based_rdirs_get_flow_direction_for_ocean_point(this) &
        result(flow_direction_for_ocean_point)
        class(latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
            allocate(flow_direction_for_ocean_point,source=&
                dir_based_direction_indicator(this%flow_direction_for_ocean_point))
    end function latlon_dir_based_rdirs_get_flow_direction_for_ocean_point

    function dir_based_rdirs_is_diagonal(this,coords_in) result(diagonal)
        class(latlon_dir_based_rdirs_cell),intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: rdir
        logical :: diagonal
            rdir => this%river_directions%get_value(coords_in)
            select type (rdir)
                type is (integer)
                if (rdir == 7 .or. rdir == 9 .or. rdir == 1 .or. rdir == 3) then
                    diagonal = .TRUE.
                else
                    diagonal = .FALSE.
                end if
            end select
            deallocate(rdir)
    end function dir_based_rdirs_is_diagonal

    subroutine latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points(this)
        class(latlon_dir_based_rdirs_cell), intent(inout) :: this
        class(*), pointer :: rdir
        logical :: non_ocean_cell_found
        integer :: i,j
            non_ocean_cell_found = .FALSE.
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    rdir => this%river_directions%get_value(latlon_coords(i,j))
                    select type (rdir)
                    type is (integer)
                        if( rdir == 0 ) then
                            this%contains_river_mouths = .TRUE.
                            this%ocean_cell = .FALSE.
                        else if( rdir /= -1 ) then
                            non_ocean_cell_found = .TRUE.
                        end if
                    end select
                    deallocate(rdir)
                    if(this%contains_river_mouths) return
                end do
            end do
            this%contains_river_mouths = .FALSE.
            this%ocean_cell = .not. non_ocean_cell_found
    end subroutine

    subroutine neighborhood_destructor(this)
        class(neighborhood), intent(inout) :: this
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
    end subroutine neighborhood_destructor

    function find_downstream_cell(this,initial_outlet_pixel) result(downstream_cell)
        class(neighborhood), intent(in) :: this
        class(coords), intent(in) :: initial_outlet_pixel
        class(coords), allocatable :: working_pixel
        integer :: downstream_cell
        class(*), pointer :: initial_cumulative_flow
        class(*), pointer :: last_cell_cumulative_flow
        logical :: coords_not_in_neighborhood
        logical :: outflow_pixel_reached
            initial_cumulative_flow => this%total_cumulative_flow%get_value(initial_outlet_pixel)
            coords_not_in_neighborhood = .FALSE.
            allocate(working_pixel,source=initial_outlet_pixel)
            call this%find_next_pixel_downstream(this,working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
            do
                downstream_cell = this%in_which_cell(working_pixel)
                do
                    last_cell_cumulative_flow => this%total_cumulative_flow%get_value(working_pixel)
                    call this%find_next_pixel_downstream(this,working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                    if (coords_not_in_neighborhood .or. .not. downstream_cell == this%in_which_cell(working_pixel) .or. &
                        outflow_pixel_reached) exit
                    deallocate(last_cell_cumulative_flow)
                end do
                if (coords_not_in_neighborhood .or. outflow_pixel_reached) exit
                select type(last_cell_cumulative_flow)
                type is (integer)
                    select type(initial_cumulative_flow)
                    type is (integer)
                        if ((last_cell_cumulative_flow  - initial_cumulative_flow > area_threshold) .and. &
                            ( .not. this%is_center_cell(downstream_cell))) exit
                    end select
                end select
                deallocate(last_cell_cumulative_flow)
            end do
            deallocate(last_cell_cumulative_flow)
            deallocate(initial_cumulative_flow)
            deallocate(working_pixel)
    end function find_downstream_cell

    function path_reenters_region(this,coords_inout)
        class(neighborhood), intent(in) :: this
        class(coords), allocatable, intent(inout) :: coords_inout
        logical path_reenters_region
        logical :: coords_not_in_neighborhood
        do
            call this%find_next_pixel_upstream(coords_inout,coords_not_in_neighborhood)
            if (coords_not_in_neighborhood) then
                path_reenters_region = .FALSE.
                exit
            else if (this%in_which_cell(coords_inout) == this%center_cell_id) then
                path_reenters_region = .TRUE.
                exit
            end if
        end do
    end function path_reenters_region

    pure function is_center_cell(this,cell_id)
        implicit none
        class(neighborhood), intent(in) :: this
        integer, intent(in) :: cell_id
        logical :: is_center_cell
            if (cell_id == this%center_cell_id) then
                is_center_cell = .TRUE.
            else
                is_center_cell = .FALSE.
            end if
    end function is_center_cell

    subroutine init_latlon_neighborhood(this,center_cell_section_coords,river_directions,&
                                        total_cumulative_flow)
        class(latlon_neighborhood) :: this
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: section_coords
        integer, target, dimension(:,:) :: river_directions
        integer, target, dimension(:,:) :: total_cumulative_flow
        class (*), pointer, dimension(:,:) :: river_directions_pointer
        class (*), pointer, dimension(:,:) :: total_cumulative_flow_pointer
            this%center_cell_min_lat = center_cell_section_coords%section_min_lat
            this%center_cell_min_lon = center_cell_section_coords%section_min_lon
            this%center_cell_max_lat =  center_cell_section_coords%section_min_lat + &
                                        center_cell_section_coords%section_width_lat - 1
            this%center_cell_max_lon =  center_cell_section_coords%section_min_lon + &
                                        center_cell_section_coords%section_width_lon - 1
            this%section_min_lat = center_cell_section_coords%section_min_lat - &
                                   center_cell_section_coords%section_width_lat
            this%section_min_lon = center_cell_section_coords%section_min_lon - &
                                   center_cell_section_coords%section_width_lon
            this%section_max_lat = this%center_cell_max_lat + &
                                   center_cell_section_coords%section_width_lat
            this%section_max_lon = this%center_cell_max_lon + &
                                   center_cell_section_coords%section_width_lon
            section_coords = latlon_section_coords(this%section_min_lat,this%section_min_lon,&
                                                   this%section_max_lat,this%section_max_lon)
            river_directions_pointer => river_directions
            this%river_directions => latlon_field_section(river_directions_pointer,&
                                                          section_coords)
            total_cumulative_flow_pointer => total_cumulative_flow
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,&
                                                               section_coords)
            this%center_cell_id = 5
    end subroutine init_latlon_neighborhood

    pure function latlon_in_which_cell(this,pixel) result (in_cell)
        class(latlon_neighborhood), intent(in) :: this
        class(coords), intent(in) :: pixel
        integer :: in_cell
        integer :: internal_cell_lat
        integer :: internal_cell_lon
            select type(pixel)
            class is (latlon_coords)
                if (pixel%lat < this%center_cell_min_lat) then
                    internal_cell_lat = 1
                else if (pixel%lat > this%center_cell_max_lat) then
                    internal_cell_lat = 3
                else
                    internal_cell_lat = 2
                end if
                if (pixel%lon < this%center_cell_min_lon) then
                    internal_cell_lon = 1
                else if (pixel%lon > this%center_cell_max_lon) then
                    internal_cell_lon = 3
                else
                    internal_cell_lon = 2
                end if
                in_cell = internal_cell_lon + 9 - 3*internal_cell_lat
            end select
    end function latlon_in_which_cell

    function latlon_dir_based_rdirs_field_constructor(field_section_coords,river_directions) &
        result(constructor)
        type(latlon_dir_based_rdirs_field), allocatable :: constructor
        type(latlon_section_coords) :: field_section_coords
        integer, dimension(:,:), target :: river_directions
        class(*), dimension(:,:), pointer :: river_directions_pointer
            allocate(constructor)
            river_directions_pointer => river_directions
            call constructor%init_latlon_field(field_section_coords,river_directions_pointer)
    end function

    function latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,river_directions,&
                                                               total_cumulative_flow) result(constructor)
        type(latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(latlon_section_coords) :: center_cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            allocate(constructor)
            call constructor%init_latlon_neighborhood(center_cell_section_coords,river_directions,&
                                                      total_cumulative_flow)
    end function latlon_dir_based_rdirs_neighborhood_constructor

    subroutine latlon_dir_based_rdirs_find_next_pixel_downstream(this,coords_inout,coords_not_in_area,&
                                                                 outflow_pixel_reached)
        class(area), intent(in) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
        class(*), pointer :: rdir
        select type (coords_inout)
        type is (latlon_coords)
            rdir => this%river_directions%get_value(coords_inout)
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
                    coords_not_in_area = .FALSE.
                    if (present(outflow_pixel_reached)) outflow_pixel_reached =  .TRUE.
                else
                    if (present(outflow_pixel_reached)) outflow_pixel_reached =  .FALSE.
                    if (coords_inout%lon > this%section_max_lon .or. coords_inout%lon < this%section_min_lon  .or. &
                        coords_inout%lat > this%section_max_lat .or. coords_inout%lat < this%section_min_lat) then
                        coords_not_in_area = .TRUE.
                    else
                        coords_not_in_area = .FALSE.
                    end if
                end if
            end select
            deallocate(rdir)
        end select
    end subroutine latlon_dir_based_rdirs_find_next_pixel_downstream

    pure function dir_based_rdirs_calculate_direction_indicator(this,downstream_cell) result(flow_direction)
        class(latlon_dir_based_rdirs_neighborhood), intent(in) :: this
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
            !Would be better to assign this directly but has not yet been implemented
            !in all Fortran compilers
            allocate(flow_direction,source=dir_based_direction_indicator(downstream_cell))
            !Pointless test to get rid of compiler warning that 'this' is unused
            if(this%section_min_lat /= 0) continue
    end function dir_based_rdirs_calculate_direction_indicator

end module area_mod
