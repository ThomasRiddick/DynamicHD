module l2_lake_model_mod

use latlon_lake_model_tree_mod

implicit none

!DELETE THIS
!AND THIS TOO
! TO DO
! debug printout etc
! basins
! lake pointers func
! lake pointers alloc
! clean up excess brackets in if statements
! systematically ues number_of_cells_when_filled
! remove doubling up of vars in constructor
! check loop ordering is optimal

! use latlon_lake_model_tree_mod
! #ifdef USE_LOGGING
!   use latlon_lake_logger_mod
! #endif

!##############################################################################
!##############################################################################
!                Section 0: Parameter definitions
!##############################################################################
!##############################################################################

integer,parameter :: dp = selected_real_kind(12)
integer, parameter :: filling_lake_type = 1
integer, parameter :: overflowing_lake_type = 2
integer, parameter :: subsumed_lake_type = 3
integer, parameter :: connect_height = 1
integer, parameter :: flood_height = 2

!##############################################################################
!##############################################################################
!                Section 1: Type Definitions and Interfaces
!##############################################################################
!##############################################################################

!An immutable (once initialised) integer keyed dictionary of redirects
type :: redirectdictionary
  integer, dimension(:), allocatable :: keys
  type(redirect), dimension(:), allocatable :: values
  integer :: number_of_entries
  integer :: next_entry_index
  logical :: initialisation_complete
end type redirectdictionary

interface redirectdictionary
  procedure :: redirectdictionaryconstructor
end interface

type :: coordslist
  _DEF_INDICES_LIST_INDEX_NAME_coords_
end type coordslist

interface coordslist
  procedure :: coordslistconstructor
end interface

type :: integerlist
  integer, pointer, dimension(:) :: list
end type

interface integerlist
  procedure :: integerlistconstructor
end interface

type :: lakemodelparameters
  _DEF_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_
  _DEF_NPOINTS_LAKE_
  _DEF_NPOINTS_HD_
  _DEF_NPOINTS_SURFACE_
  real(dp) :: lake_retention_constant
  real(dp) :: minimum_lake_volume_threshold
  integer :: number_of_lakes
  type(integerlist), pointer, dimension(:) :: basins
  integer, pointer, dimension(_DIMS_) :: basin_numbers
  _DEF_INDICES_LIST_cells_with_lakes_INDEX_NAME_
  real(dp), pointer, dimension(_DIMS_) :: cell_areas_on_surface_model_grid
  logical, pointer, dimension(_DIMS_) :: lake_centers
  integer, pointer, dimension(_DIMS_) :: number_fine_grid_cells
  type(coordslist), pointer, dimension(:) :: surface_cell_to_fine_cell_maps
  integer, pointer, dimension(_DIMS_) :: surface_cell_to_fine_cell_map_numbers
end type lakemodelparameters

interface lakemodelparameters
  procedure :: lakemodelparametersconstructor
end interface lakemodelparameters

type :: lakemodelprognostics
  type(lakepointer), pointer, dimension(:) :: lakes
  integer, pointer, dimension(_DIMS_) :: lake_numbers
  integer, pointer, dimension(_DIMS_) :: lake_cell_count
  real(dp) :: total_lake_volume
  real(dp), pointer, dimension(_DIMS_) :: effective_volume_per_cell_on_surface_grid
  real(dp), pointer, dimension(_DIMS_) :: effective_lake_height_on_surface_grid_to_lakes
  real(dp), pointer, dimension(_DIMS_) :: new_effective_volume_per_cell_on_surface_grid
  real(dp), pointer, dimension(_DIMS_) :: evaporation_on_surface_grid
  real(dp), pointer, dimension(_DIMS_) :: water_to_lakes
  real(dp), pointer, dimension(_DIMS_) :: water_to_hd
  real(dp), pointer, dimension(_DIMS_) :: lake_water_from_ocean
  real(dp), pointer, dimension(:) :: evaporation_from_lakes
  logical, pointer, dimension(:) :: evaporation_applied
  type(rooted_tree_forest), pointer :: set_forest
end type lakemodelprognostics

interface lakemodelprognostics
  procedure :: lakemodelprognosticsconstructor
end interface lakemodelprognostics

type :: cell
  _DEF_COORDS_coords_
  integer :: height_type
  real(dp) :: fill_threshold
  real(dp) :: height
end type cell

interface cell
  procedure :: cellconstructor
end interface cell

type :: redirect
  logical :: use_local_redirect
  integer :: local_redirect_target_lake_number
  _DEF_COORDS_non_local_redirect_target_
end type redirect

interface redirect
  procedure :: redirectconstructor
end interface redirect

type :: lakeparameters
  _DEF_COORDS_center_coords_
  _DEF_COORDS_center_cell_coarse_coords_
  integer :: lake_number
  logical :: is_primary
  logical :: is_leaf
  integer :: primary_lake
  integer :: number_of_cells_when_filled
  integer(dp), allocatable, dimension(:) :: secondary_lakes
  type(cell), allocatable, dimension(:) :: filling_order
  type(redirectdictionary), allocatable :: outflow_points
end type lakeparameters

interface lakeparameters
  procedure :: lakeparametersconstructor
end interface lakeparameters

type :: lakeprognostics
  !Common variables
  integer :: lake_type
  real(dp) :: unprocessed_water
  logical :: active_lake
  type(lakeparameters), pointer :: parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  ! Filling lake variables
  _DEF_COORDS_current_cell_to_fill_
  integer :: current_height_type
  integer :: current_filling_cell_index
  real(dp) :: next_cell_volume_threshold
  real(dp) :: previous_cell_volume_threshold
  real(dp) :: lake_volume
  ! Overflowing lake variables
  type(redirect),pointer :: current_redirect
  real(dp) :: excess_water
  ! Subsumed lake variables
  integer :: redirect_target
end type lakeprognostics

interface lakeprognostics
  procedure :: lakeprognosticsconstructor
end interface lakeprognostics

type :: lakepointer
  type(lakeprognostics), pointer :: lake_pointer
end type lakepointer

type :: lakeparameterspointer
  type(lakeparameters), pointer :: lake_parameters_pointer
end type lakeparameterspointer

contains

!##############################################################################
!##############################################################################
!              Section 2: Constructors and Initialisation Routines
!##############################################################################
!##############################################################################

function redirectdictionaryconstructor(number_of_entries) result(constructor)
  integer, intent(in) :: number_of_entries
  type(redirectdictionary) :: constructor
    allocate(constructor%keys(number_of_entries))
    allocate(constructor%values(number_of_entries))
    constructor%initialisation_complete = .false.
    constructor%next_entry_index = 1
end function redirectdictionaryconstructor

function coordslistconstructor(_INDICES_LIST_INDEX_NAME_coords_) &
    result(constructor)
  _DEF_INDICES_LIST_INDEX_NAME_coords_
  type(coordslist) :: constructor
    _ASSIGN_constructor%_INDICES_LIST_INDEX_NAME_coords_ = &
      _INDICES_LIST_INDEX_NAME_coords_
end function coordslistconstructor

function integerlistconstructor(list) result(constructor)
  integer, dimension(:), intent(in) :: list
  type(integerlist) :: constructor
    constructor%list = list
end function integerlistconstructor

function lakemodelparametersconstructor( &
    _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
    cell_areas_on_surface_model_grid, &
    number_of_lakes, &
    is_lake, &
    _NPOINTS_HD_, &
    _NPOINTS_LAKE_, &
    _NPOINTS_SURFACE_, &
    lake_retention_constant, &
    minimum_lake_volume_threshold) result(constructor)
  integer, intent(in) :: number_of_lakes
  logical, allocatable, dimension(_DIMS_), intent(in) :: is_lake
  _DEF_NPOINTS_HD_ _INTENT_in_
  _DEF_NPOINTS_LAKE_ _INTENT_in_
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
  _DEF_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _INTENT_in_
  real(dp), pointer, dimension(_DIMS_), intent(in) :: cell_areas_on_surface_model_grid
  real(dp), intent(inout), optional :: lake_retention_constant
  real(dp), intent(inout), optional :: minimum_lake_volume_threshold
  type(lakemodelparameters), pointer :: constructor
  type(integerlist), pointer, dimension(:) :: basins
  integer, pointer, dimension(_DIMS_) :: basin_numbers
  integer, allocatable, dimension(_DIMS_) :: number_fine_grid_cells
  _DEF_INDICES_LIST_cells_with_lakes_INDEX_NAME_
  _DEF_INDICES_LIST_surface_cell_to_fine_cell_map_INDEX_NAMEs_temp_
  integer, pointer, dimension(_DIMS_) :: number_of_lake_cells_temp
  logical, pointer, dimension(_DIMS_) :: needs_map
  type(coordslist), pointer, dimension(:) :: surface_cell_to_fine_cell_maps
  integer, pointer, dimension(_DIMS_) :: surface_cell_to_fine_cell_map_numbers
  _DEF_COORDS_surface_model_
  integer :: number_of_lake_cells
  integer :: surface_cell_to_fine_cell_map_number
  integer :: map_number
  integer :: k
  _DEF_LOOP_INDEX_LAKE_
  _DEF_LOOP_INDEX_SURFACE_
    allocate(constructor)
    if (.not. present(lake_retention_constant)) then
      lake_retention_constant = 0.1_dp
    end if
    if (.not. present(minimum_lake_volume_threshold)) then
      minimum_lake_volume_threshold = 0.0000001_dp
    end if
    allocate(number_fine_grid_cells(_NPOINTS_SURFACE_))
    number_fine_grid_cells(_DIMS_) = 0
    allocate(needs_map(_NPOINTS_SURFACE_))
    needs_map(_DIMS_) = .false.
    allocate(number_of_lake_cells_temp(_NPOINTS_SURFACE_))
    number_of_lake_cells_temp(_DIMS_) = 0
    _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
      _GET_COORDS_ _COORDS_surface_model_ _FROM_ _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _COORDS_LAKE_
      number_fine_grid_cells(_COORDS_ARG_surface_model_) = &
           number_fine_grid_cells(_COORDS_ARG_surface_model_) + 1
      if (is_lake(_COORDS_LAKE_)) then
          needs_map(_COORDS_ARG_surface_model_) = .true.
          number_of_lake_cells_temp(_COORDS_ARG_surface_model_) = &
            number_of_lake_cells_temp(_COORDS_ARG_surface_model_) + 1
      end if
    _LOOP_OVER_LAKE_GRID_END_
    allocate(surface_cell_to_fine_cell_map_numbers(_NPOINTS_SURFACE_))
    surface_cell_to_fine_cell_map_numbers(_DIMS_) = 0
    map_number = 0
    _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
      if (needs_map(_COORDS_SURFACE_)) then
        map_number = map_number + 1
      end if
    _LOOP_OVER_SURFACE_GRID_END_
    allocate(surface_cell_to_fine_cell_maps(map_number))
    map_number = 0
    _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
      if (needs_map(_COORDS_SURFACE_)) then
        map_number = map_number + 1
        number_of_lake_cells = number_of_lake_cells_temp(_COORDS_SURFACE_)
        allocate(_INDICES_LIST_surface_cell_to_fine_cell_map_INDEX_NAMEs_temp(number_of_lake_cells)_)
        _ASSIGN_INDICES_LIST_surface_cell_to_fine_cell_map_INDEX_NAMEs_temp(:)_ = -1
        surface_cell_to_fine_cell_maps(map_number) = &
          coordslist(_INDICES_LIST_surface_cell_to_fine_cell_map_INDEX_NAMEs_temp_)
        surface_cell_to_fine_cell_map_numbers(_COORDS_SURFACE_) = map_number
      end if
    _LOOP_OVER_SURFACE_GRID_END_
    deallocate(number_of_lake_cells_temp)
    _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
      if (is_lake(_COORDS_LAKE_)) then
        _GET_COORDS_ _COORDS_surface_model_ _FROM_ _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _COORDS_LAKE_
        surface_cell_to_fine_cell_map_number = &
          surface_cell_to_fine_cell_map_numbers(_COORDS_ARG_surface_model_)
        k = 1
        do while (surface_cell_to_fine_cell_maps(&
                  surface_cell_to_fine_cell_map_number)%&
                  &_INDICES_LIST_INDEX_NAME_coords_FIRST_DIM_(k) /= -1)
          k = k + 1
        end do
        _ASSIGN_surface_cell_to_fine_cell_maps(&
                      surface_cell_to_fine_cell_map_number)%_INDICES_LIST_INDEX_NAME_coords(k)_ = _COORDS_surface_model_
      end if
    _LOOP_OVER_LAKE_GRID_END_
    deallocate(needs_map)
    constructor%lake_retention_constant = lake_retention_constant
    constructor%minimum_lake_volume_threshold = minimum_lake_volume_threshold
    _ASSIGN_constructor%_NPOINTS_LAKE_ = _NPOINTS_LAKE_
    _ASSIGN_constructor%_NPOINTS_HD_ = _NPOINTS_HD_
    _ASSIGN_constructor%_NPOINTS_SURFACE_ = _NPOINTS_SURFACE_
    constructor%lake_retention_constant = lake_retention_constant
    constructor%minimum_lake_volume_threshold = minimum_lake_volume_threshold
    constructor%number_of_lakes = number_of_lakes
    constructor%basins => null()
    constructor%basin_numbers(_DIMS_) = 0
    _ASSIGN_constructor%_INDICES_LIST_cells_with_lakes_INDEX_NAME_ = _INDICES_LIST_cells_with_lakes_INDEX_NAME_
    constructor%cell_areas_on_surface_model_grid = cell_areas_on_surface_model_grid
    constructor%lake_centers(_DIMS_) = .false.
    constructor%number_fine_grid_cells(_DIMS_) = number_fine_grid_cells
    constructor%surface_cell_to_fine_cell_maps = surface_cell_to_fine_cell_maps
    constructor%surface_cell_to_fine_cell_map_numbers = surface_cell_to_fine_cell_map_numbers
end function lakemodelparametersconstructor

function lakemodelprognosticsconstructor(lake_model_parameters) result(constructor)
    type(lakemodelparameters), intent(in) :: lake_model_parameters
    type(lakemodelprognostics), pointer :: constructor
      allocate(constructor)
      allocate(constructor%lakes(lake_model_parameters%number_of_lakes))
      constructor%lake_numbers(_DIMS_) = 0
      constructor%lake_cell_count(_DIMS_) = 0
      constructor%effective_volume_per_cell_on_surface_grid(_DIMS_) = 0.0_dp
      constructor%effective_lake_height_on_surface_grid_to_lakes(_DIMS_) = 0.0_dp
      constructor%new_effective_volume_per_cell_on_surface_grid(_DIMS_) = 0.0_dp
      constructor%evaporation_on_surface_grid(_DIMS_) = 0.0_dp
      constructor%water_to_lakes(_DIMS_)= 0.0_dp
      constructor%water_to_hd(_DIMS_) = 0.0_dp
      constructor%lake_water_from_ocean(_DIMS_) = 0.0_dp
      allocate(constructor%evaporation_from_lakes(lake_model_parameters%number_of_lakes))
      constructor%evaporation_from_lakes(:) = 0.0_dp
      allocate(constructor%evaporation_applied(lake_model_parameters%number_of_lakes))
      constructor%evaporation_applied(:) = .false.
      constructor%set_forest => rooted_tree_forest()
end function lakemodelprognosticsconstructor

function cellconstructor(_COORDS_ARG_coords_, &
                         height_type, &
                         fill_threshold, &
                         height) result(constructor)
  _DEF_COORDS_coords_ _INTENT_in_
  integer  :: height_type
  real(dp) :: fill_threshold
  real(dp) :: height
  type(cell), pointer :: constructor
    _ASSIGN_constructor%_COORDS_coords_ = _COORDS_coords_
    constructor%height_type = height_type
    constructor%fill_threshold = fill_threshold
    constructor%height = height
end function cellconstructor

function redirectconstructor(use_local_redirect, &
                             local_redirect_target_lake_number, &
                             _COORDS_ARG_non_local_redirect_target_) &
                             result(constructor)
  logical, intent(in) :: use_local_redirect
  integer, intent(in) :: local_redirect_target_lake_number
  _DEF_COORDS_non_local_redirect_target_ _INTENT_in_
  type(redirect), pointer :: constructor
    constructor%use_local_redirect = use_local_redirect
    constructor%local_redirect_target_lake_number = &
      local_redirect_target_lake_number
    _ASSIGN_constructor%_COORDS_non_local_redirect_target_ = &
      _COORDS_non_local_redirect_target_
end function redirectconstructor

function lakeparametersconstructor(lake_number, &
                                   primary_lake, &
                                   secondary_lakes, &
                                   _COORDS_ARG_center_coords_, &
                                   filling_order, &
                                   outflow_points, &
                                   _NPOINTS_LAKE_, &
                                   _NPOINTS_HD_) &
                                   result(constructor)
  integer, intent(in) :: lake_number
  integer, intent(in) :: primary_lake
  integer, dimension(:), allocatable, intent(in) :: secondary_lakes
  _DEF_COORDS_center_coords_ _INTENT_in_
  type(cell), allocatable, dimension(:), intent(in) :: filling_order
  type(redirectdictionary), intent(in) :: outflow_points
  _DEF_NPOINTS_LAKE_ _INTENT_in_
  _DEF_NPOINTS_HD_ _INTENT_in_
  _DEF_COORDS_fine_cells_per_coarse_cell_
  type(lakeparameters), pointer :: constructor
  logical :: is_primary
  logical :: is_leaf
    allocate(constructor)
    _ASSIGN_constructor%_COORDS_center_coords_ = _COORDS_center_coords_
    _GET_COARSE_COORDS_FROM_FINE_COORDS_ constructor%_COORDS_center_cell_coarse_coords_ _NPOINTS_LAKE_ _NPOINTS_HD_ _COORDS_center_coords_
    constructor%lake_number = lake_number
    constructor%is_primary = (primary_lake == -1)
    if (is_primary .and. get_dictionary_size(outflow_points) > 1) then
      write(*,*) "Primary lake has more than one outflow point"
      stop
    end if
    constructor%is_leaf = (size(secondary_lakes) == 0)
    constructor%primary_lake = primary_lake
    constructor%secondary_lakes = secondary_lakes
    constructor%filling_order = filling_order
    constructor%outflow_points = outflow_points
    constructor%number_of_cells_when_filled = size(filling_order)
end function lakeparametersconstructor

!Construct an empty filling lake by default
function lakeprognosticsconstructor(lake_parameters, &
                                    lake_model_parameters, &
                                    lake_model_prognostics) result(constructor)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(in) :: lake_model_prognostics
  type(lakeprognostics), pointer :: constructor
    allocate(constructor)
    constructor%lake_type = filling_lake_type
    constructor%active_lake = lake_parameters%is_leaf
    constructor%unprocessed_water = 0.0_dp
    constructor%parameters = lake_parameters
    constructor%lake_model_parameters = lake_model_parameters
    constructor%lake_model_prognostics = lake_model_prognostics
    call initialise_filling_lake(constructor,.false.)
end function lakeprognosticsconstructor

subroutine initialise_filling_lake(lake,initialise_filled)
  type(lakeprognostics), pointer, intent(inout) :: lake
  logical, intent(in) :: initialise_filled
  type(lakeparameters), pointer :: lake_parameters
  integer :: filling_order_length
    lake_parameters => lake%parameters
    call set_overflowing_lake_parameter_to_null_values(lake)
    call set_subsumed_lake_parameter_to_null_values(lake)
    if (initialise_filled) then
      filling_order_length = size(lake_parameters%filling_order)
      _ASSIGN_lake%_COORDS_current_cell_to_fill_ = &
        lake_parameters%filling_order(filling_order_length)%_COORDS_coords_
      lake%current_height_type = lake_parameters%filling_order(filling_order_length)%height_type
      lake%current_filling_cell_index = filling_order_length
      lake%next_cell_volume_threshold = &
        lake_parameters%filling_order(filling_order_length)%fill_threshold
      if (size(lake_parameters%filling_order) > 1) then
        lake%previous_cell_volume_threshold = &
          lake_parameters%filling_order(filling_order_length-1)%fill_threshold
      else
        lake%previous_cell_volume_threshold = 0.0_dp
      end if
      lake%lake_volume = lake_parameters%filling_order(filling_order_length)%fill_threshold
    else
      _ASSIGN_lake%_COORDS_current_cell_to_fill_ = &
        lake_parameters%filling_order(1)%_COORDS_coords_
      lake%current_height_type = &
        lake_parameters%filling_order(1)%height_type
      lake%current_filling_cell_index = 1
      lake%next_cell_volume_threshold = &
        lake_parameters%filling_order(1)%fill_threshold
      lake%previous_cell_volume_threshold = 0.0_dp
      lake%lake_volume = 0.0_dp
    end if
end subroutine initialise_filling_lake

subroutine initialise_overflowing_lake(lake, &
                                       current_redirect, &
                                       excess_water)
  type(redirect), pointer, intent(in) :: current_redirect
  real(dp), intent(in) :: excess_water
  type(lakeprognostics), pointer, intent(inout) :: lake
    call set_filling_lake_parameter_to_null_values(lake)
    call set_subsumed_lake_parameter_to_null_values(lake)
    lake%current_redirect = current_redirect
    lake%excess_water = excess_water
end subroutine initialise_overflowing_lake

subroutine initialise_subsumed_lake(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
    call set_filling_lake_parameter_to_null_values(lake)
    call set_overflowing_lake_parameter_to_null_values(lake)
    lake%redirect_target = -1
end subroutine initialise_subsumed_lake

!##############################################################################
!##############################################################################
!                  Section 3: Utility subprograms
!##############################################################################
!##############################################################################

subroutine add_entry_to_dictionary(dict,key,value)
  type(redirectdictionary) :: dict
  integer :: key
  type(redirect) :: value
    if (dict%initialisation_complete) then
      write(*,*) "Error - trying to write to finished dictionary"
      stop
    end if
    dict%keys(dict%next_entry_index) = key
    dict%values(dict%next_entry_index) = value
    dict%next_entry_index = dict%next_entry_index + 1
end subroutine add_entry_to_dictionary

subroutine finish_dictionary(dict)
  type(redirectdictionary), intent(inout) :: dict
  integer, dimension(:), allocatable :: sorted_keys
  type(redirect), dimension(:), allocatable :: sorted_values
  integer :: max_key_value
  integer, dimension(1) :: minimum_location_array
  integer :: minimum_location
  integer :: number_of_entries
  integer :: i
    dict%number_of_entries = size(dict%keys)
    max_key_value = maxval(dict%keys)
    allocate(sorted_keys(dict%number_of_entries))
    allocate(sorted_values(dict%number_of_entries))
    do i = 1,number_of_entries
      minimum_location_array = minloc(dict%keys)
      minimum_location = minimum_location_array(1)
      sorted_keys(i) = dict%keys(minimum_location)
      sorted_values(i) = dict%values(minimum_location)
      dict%keys(i) = max_key_value + 1
    end do
    dict%keys(:) = sorted_keys(:)
    dict%values(:) = sorted_values(:)
    dict%initialisation_complete = .true.
end subroutine finish_dictionary

function get_dictionary_entry(dict,key) result(value)
  type(redirectdictionary), intent(in) :: dict
  integer, intent(in) :: key
  type(redirect), pointer :: value
  integer :: section_start_index
  integer :: section_end_index
  integer :: section_length
  integer :: midpoint_index
  integer :: midpoint_key_value
  integer :: i
    if (.not. dict%initialisation_complete) then
      write(*,*) "Error - trying to read from unfinished dictionary"
      stop
    end if
    section_start_index = 1
    section_end_index = dict%number_of_entries
    section_length  = dict%number_of_entries
    do
      if (section_length > 4) then
        midpoint_index = section_length/2 + section_start_index
        midpoint_key_value = dict%keys(midpoint_index)
        if (midpoint_key_value == key) then
            value = dict%values(midpoint_index)
            return
        else if (midpoint_key_value > key) then
          section_end_index = midpoint_index - 1
          section_length  = section_end_index + 1 - section_start_index
        else
          section_start_index = midpoint_index + 1
          section_length  = section_end_index + 1 - section_start_index
        end if
      else
        do i = section_start_index,section_end_index
          if (dict%keys(i) == key) then
            value = dict%values(i)
            return
          end if
        end do
      end if
    end do
end function get_dictionary_entry

function get_dictionary_size(dict) result(number_of_entries)
  type(redirectdictionary), intent(in) :: dict
  integer :: number_of_entries
    number_of_entries = dict%number_of_entries
end function get_dictionary_size

! !##############################################################################
! !##############################################################################
! !                  Section 4: Subprograms on individual lakes
! !##############################################################################
! !##############################################################################

recursive subroutine add_water(lake,inflow,store_water)
  type(lakeprognostics), pointer, intent(inout) :: lake
  real(dp), intent(in) :: inflow
  logical, intent(in) :: store_water
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  type(lakeprognostics), pointer :: sublake
  type(lakeprognostics), pointer :: other_lake
  _DEF_COORDS_surface_model_coords_
  real(dp) :: inflow_local
  real(dp) :: new_lake_volume
  integer :: other_lake_number
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    inflow_local = inflow
    if (lake%lake_type == filling_lake_type) then
      if (.not. lake%active_lake) then
        !Take the first sub-basin of this lake
        sublake => lake_model_prognostics%lakes(lake_parameters%secondary_lakes(1))%lake_pointer
        call add_water(sublake,inflow_local,.false.)
        return
      end if
      if (store_water) then
        lake%unprocessed_water = lake%unprocessed_water + inflow_local
        return
      end if
      do while (inflow_local > 0.0_dp)
        if (lake%lake_volume >= 0 .and. lake%current_filling_cell_index == 1 .and. &
            lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) == 0 .and. &
            .not. lake_parameters%is_leaf .and. lake%current_height_type == flood_height) then
          _GET_COORDS_ _COORDS_surface_model_coords_ _FROM_ lake_model_parameters%_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ lake%_COORDS_current_cell_to_fill_
          lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) = &
            lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) + 1
          lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) = &
            lake%parameters%lake_number
        end if
        new_lake_volume = inflow_local + lake%lake_volume
        if (new_lake_volume <= lake%next_cell_volume_threshold) then
          lake%lake_volume = new_lake_volume
          inflow_local = 0.0_dp
        else
          inflow_local = new_lake_volume - lake%next_cell_volume_threshold
          lake%lake_volume = lake%next_cell_volume_threshold
          if (lake%current_filling_cell_index == size(lake%parameters%filling_order)) then
            if (check_if_merge_is_possible(lake_parameters,lake_model_prognostics)) then
              call merge_lakes(inflow_local,lake_parameters, &
                               lake_model_prognostics)
            else
              call change_to_overflowing_lake(lake,inflow_local)
            end if
            return
          end if
          lake%current_filling_cell_index = lake%current_filling_cell_index + 1
          lake%previous_cell_volume_threshold = lake%next_cell_volume_threshold
          lake%next_cell_volume_threshold = &
            lake%parameters%filling_order(lake%current_filling_cell_index)%fill_threshold
          _ASSIGN_lake%_COORDS_current_cell_to_fill_ = &
            lake%parameters%filling_order(lake%current_filling_cell_index)%_COORDS_coords_
          lake%current_height_type = &
            lake%parameters%filling_order(lake%current_filling_cell_index)%height_type
          if (lake%current_height_type == flood_height .and. &
              lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) == 0) then
          _GET_COORDS_ _COORDS_surface_model_coords_ _FROM_ lake_model_parameters%_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ lake%_COORDS_current_cell_to_fill_
            lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) = &
              lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) + 1
            lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) = &
              lake%parameters%lake_number
          end if
        end if
      end do
    else if (lake%lake_type == overflowing_lake_type) then
      if (.not. lake%active_lake) then
        write(*,*) "Water added to inactive lake"
        stop
      end if
      if (store_water) then
        lake%unprocessed_water = lake%unprocessed_water + inflow_local
        return
      end if
      if (lake%current_redirect%use_local_redirect) then
        other_lake_number = lake%current_redirect%local_redirect_target_lake_number
        other_lake => &
          lake_model_prognostics%lakes(other_lake_number)%lake_pointer
        call add_water(other_lake,inflow_local,.false.)
      else
        lake%excess_water = lake%excess_water + inflow_local
      end if
    else if (lake%lake_type == subsumed_lake_type) then
      if (.not. lake%active_lake) then
        write(*,*) "Water added to inactive lake"
        stop
      end if
      if (lake%redirect_target == -1) then
        lake%redirect_target = &
          lake%lake_model_prognostics%set_forest%find_root_from_label(&
            lake_parameters%lake_number)
      end if
      call add_water(lake_model_prognostics%lakes(lake%redirect_target)%lake_pointer, &
                     inflow_local,store_water)
    else
      write(*,*) "Unrecognised lake type"
      stop
    end if
end subroutine add_water

recursive subroutine remove_water(lake,outflow,store_water)
  type(lakeprognostics), pointer, intent(inout) :: lake
  real(dp), intent(in) :: outflow
  logical, intent(in) :: store_water
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  _DEF_COORDS_surface_model_coords_
  real(dp) :: outflow_local
  real(dp) :: new_lake_volume
  real(dp) :: minimum_new_lake_volume
  integer :: redirect_target
  logical :: repeat_loop
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    outflow_local = outflow
    if (lake%lake_type == filling_lake_type) then
      if (.not. lake%active_lake) then
        write(*,*) "Water removed from inactive lake"
        stop
      end if
      if (store_water) then
        lake%unprocessed_water = lake%unprocessed_water - outflow_local
        return
      end if
      repeat_loop = .false.
      do while (outflow_local > 0.0_dp .or. repeat_loop)
        repeat_loop = .false.
        new_lake_volume = lake%lake_volume - outflow_local
        if (new_lake_volume > 0 .and. new_lake_volume <= &
            lake_model_parameters%minimum_lake_volume_threshold .and. &
            lake_parameters%is_leaf) then
          lake_model_prognostics%lake_water_from_ocean(&
            lake_parameters%_COORDS_ARG_center_cell_coarse_coords_) = &
              lake_model_prognostics%lake_water_from_ocean(&
              lake_parameters%_COORDS_ARG_center_cell_coarse_coords_) - &
              new_lake_volume
          new_lake_volume = 0
        end if
        minimum_new_lake_volume = lake%previous_cell_volume_threshold
        if (new_lake_volume <= 0.0_dp .and. lake%current_filling_cell_index == 1) then
          outflow_local = 0.0_dp
          if (lake_parameters%is_leaf) then
            lake%lake_volume = new_lake_volume
          else
            lake%lake_volume = 0.0_dp
            call split_lake(lake,-new_lake_volume,lake_parameters, &
                            lake_model_prognostics)
          end if
          if (lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) == &
              lake%parameters%lake_number .and. &
              .not. lake_parameters%is_leaf .and. lake%current_height_type == flood_height) then
            _GET_COORDS_ _COORDS_surface_model_coords_ _FROM_ lake_model_parameters%_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ lake%_COORDS_current_cell_to_fill_
            lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) = &
              lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) - 1
            lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) = 0
          end if
        else if (new_lake_volume >= minimum_new_lake_volume .and. &
                 new_lake_volume > 0) then
          outflow_local = 0.0_dp
          lake%lake_volume = new_lake_volume
        else
          outflow_local = minimum_new_lake_volume - new_lake_volume
          lake%lake_volume = minimum_new_lake_volume
          if (lake%current_height_type == flood_height .and. &
              lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) == &
              lake%parameters%lake_number) then
            _GET_COORDS_ _COORDS_surface_model_coords_ _FROM_ lake_model_parameters%_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ lake%_COORDS_current_cell_to_fill_
            lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) = &
              lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) - 1
            lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) = 0
          end if
          lake%current_filling_cell_index = lake%current_filling_cell_index - 1
          lake%next_cell_volume_threshold = lake%previous_cell_volume_threshold
          if (lake%current_filling_cell_index > 1) then
            lake%previous_cell_volume_threshold = &
              lake%parameters%filling_order(lake%current_filling_cell_index-1)%fill_threshold
          else
            lake%previous_cell_volume_threshold = 0.0_dp
          end if
          _ASSIGN_lake%_COORDS_current_cell_to_fill_ = lake%parameters%filling_order(lake%current_filling_cell_index)%_COORDS_coords_
          lake%current_height_type = &
            lake%parameters%filling_order(lake%current_filling_cell_index)%height_type
          if (lake%current_filling_cell_index > 1) then
            repeat_loop = .true.
          end if
        end if
      end do
    else if (lake%lake_type == overflowing_lake_type) then
      if (.not. lake%active_lake) then
        write(*,*) "Water removed from inactive lake"
        stop
      end if
      if (store_water) then
        lake%unprocessed_water = lake%unprocessed_water - outflow_local
      end if
      if (outflow_local <= lake%excess_water) then
        lake%excess_water = lake%excess_water - outflow_local
      end if
      outflow_local = outflow_local - lake%excess_water
      lake%excess_water = 0.0_dp
      call change_overflowing_lake_to_filling_lake(lake)
      call remove_water(lake,outflow_local,.false.)
    else if (lake%lake_type == subsumed_lake_type) then
      !Redirect to primary
      if (.not. lake%active_lake) then
        write(*,*) "Water removed from inactive lake"
        stop
      end if
      if (lake%redirect_target == -1) then
        lake%redirect_target = lake_model_prognostics%set_forest%find_root_from_label(&
                                 lake_parameters%lake_number)
      end if
      !Copy this variables from the lake object as the lake object will
      !have its redirect set to -1 if the redirect target splits
      redirect_target = lake%redirect_target
      call remove_water(lake_model_prognostics%lakes(redirect_target)%lake_pointer, &
                        outflow_local,store_water)
    else
      write(*,*) "Unrecognised lake type"
      stop
    end if
end subroutine remove_water

subroutine process_water(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
    lake%unprocessed_water = 0.0_dp
    if (lake%unprocessed_water > 0.0_dp) then
        call add_water(lake,lake%unprocessed_water,.false.)
    else if (lake%unprocessed_water < 0.0_dp) then
        call remove_water(lake,-lake%unprocessed_water,.false.)
    end if
end subroutine process_water

function check_if_merge_is_possible(lake_parameters, &
                                    lake_model_prognostics) result(merge_possible)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  logical :: all_secondary_lakes_filled
  integer :: secondary_lake
  logical :: merge_possible
  integer :: i
    if (.not. lake_parameters%is_primary) then
      all_secondary_lakes_filled = .true.
      do i = 1,size(lake_model_prognostics%lakes(lake_parameters%primary_lake)%lake_pointer%&
                    &parameters%secondary_lakes)
        secondary_lake = lake_model_prognostics%lakes(lake_parameters%primary_lake)%lake_pointer%&
                         &parameters%secondary_lakes(i)
        if (secondary_lake /= lake_parameters%lake_number) then
          all_secondary_lakes_filled = all_secondary_lakes_filled .and. &
            (lake_model_prognostics%lakes(secondary_lake)%lake_pointer%lake_type == overflowing_lake_type)
        end if
      end do
      merge_possible = all_secondary_lakes_filled
    else
      merge_possible = .false.
    end if
end function check_if_merge_is_possible

subroutine merge_lakes(inflow, &
                       lake_parameters, &
                       lake_model_prognostics)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), intent(in) :: inflow
  type(lakeprognostics), pointer :: primary_lake
  type(lakeprognostics), pointer :: other_lake
  real(dp) :: excess_water
  real(dp) :: total_excess_water
  integer :: secondary_lake
  integer :: i
    total_excess_water = 0.0_dp
    primary_lake => lake_model_prognostics%lakes(lake_parameters%primary_lake)%lake_pointer
    primary_lake%active_lake = .true.
    do i = 1,size(primary_lake%parameters%secondary_lakes)
      secondary_lake = primary_lake%parameters%secondary_lakes(i)
      other_lake = lake_model_prognostics%lakes(secondary_lake)%lake_pointer
      if (((other_lake%lake_type == filling_lake_type) .and. &
            lake_parameters%lake_number /= secondary_lake) .or. &
           (other_lake%lake_type == subsumed_lake_type)) then
        write(*,*) "wrong lake type when merging"
        stop
      end if
      call change_to_subsumed_lake(other_lake,excess_water)
      total_excess_water = total_excess_water + excess_water
    end do
      call add_water(lake_model_prognostics%lakes(primary_lake%parameters%lake_number)%lake_pointer, &
                     inflow+total_excess_water,.false.)
end subroutine merge_lakes

subroutine split_lake(lake,water_deficit, &
                      lake_parameters, &
                      lake_model_prognostics)
  type(lakeprognostics), pointer, intent(inout) :: lake
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), intent(in) :: water_deficit
  real(dp) :: water_deficit_per_lake
  integer :: secondary_lake
  integer :: i
    lake%active_lake = .false.
    water_deficit_per_lake = water_deficit/size(lake_parameters%secondary_lakes)
    do i = 1,size(lake_parameters%secondary_lakes)
      secondary_lake = lake_parameters%secondary_lakes(i)
      call change_subsumed_to_filling_lake(lake_model_prognostics%lakes(secondary_lake)%lake_pointer, &
                                           lake_parameters%lake_number)
      call remove_water(lake_model_prognostics%lakes(secondary_lake)%lake_pointer, &
                        water_deficit_per_lake,.false.)
    end do
end subroutine split_lake

subroutine change_to_overflowing_lake(lake,inflow)
  type(lakeprognostics), pointer, intent(inout) :: lake
  real(dp), intent(in) :: inflow
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  type(redirect), pointer:: working_redirect
  integer :: secondary_lake
  logical :: redirect_target_found
  integer :: i
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    ! if debug
    !   println("Lake $(lake_parameters%lake_number) changing from filling to overflowing lake ")
    ! end
    if (.not. lake_parameters%is_primary) then
      do i = 1,size(lake_model_prognostics%lakes(lake_parameters%primary_lake)%&
                    &lake_pointer%parameters%secondary_lakes)
        secondary_lake = lake_model_prognostics%lakes(lake_parameters%primary_lake)%&
                         lake_pointer%parameters%secondary_lakes(i)
        redirect_target_found = .false.
        if ((lake_model_prognostics%lakes(secondary_lake)%lake_pointer%lake_type &
             == filling_lake_type) .and. &
            (secondary_lake /= lake_parameters%lake_number)) then
          working_redirect => &
            get_dictionary_entry(lake_parameters%outflow_points,secondary_lake)
          redirect_target_found = .true.
          exit
        end if
      end do
      if (.not. redirect_target_found) then
        write(*,*) "Non primary lake has no valid outflows"
        stop
      end if
    else
      if (lake_parameters%outflow_points%keys(1) == -1) then
        working_redirect => &
          get_dictionary_entry(lake_parameters%outflow_points,-1)
      else
        working_redirect => &
          lake_parameters%outflow_points%values(1)
      end if
    end if
    call initialise_overflowing_lake(lake, &
                                     working_redirect, &
                                     inflow)
end subroutine change_to_overflowing_lake

subroutine change_to_subsumed_lake(lake,excess_water)
  type(lakeprognostics), pointer, intent(inout) :: lake
  real(dp),intent(out) :: excess_water
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    ! if debug
    !   println("Lake $(lake_parameters%lake_number) accepting merge")
    ! end
    call initialise_subsumed_lake(lake)
    call make_new_link(lake_model_prognostics%set_forest, &
                       lake_parameters%primary_lake, &
                       lake_parameters%lake_number)
    if (lake%lake_type == overflowing_lake_type) then
      excess_water = lake%excess_water
    else
      excess_water = 0.0_dp
    end if
end subroutine change_to_subsumed_lake

subroutine change_overflowing_lake_to_filling_lake(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    if (lake%excess_water /= 0.0_dp) then
      write(*,*) "Can't change lake with excess water back to filling lake"
      stop
    end if
    ! if debug
    !   println("Lake $(lake_parameters%lake_number) changing from overflowing to filling lake ")
    ! end
    call initialise_filling_lake(lake,.true.)
end subroutine change_overflowing_lake_to_filling_lake

subroutine change_subsumed_to_filling_lake(lake,split_from_lake_number)
  type(lakeprognostics), pointer, intent(inout) :: lake
  integer, intent(in) :: split_from_lake_number
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  type(rooted_tree), pointer :: x
  type(rooted_tree), pointer :: root
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    call split_set(lake_model_prognostics%set_forest, &
                   split_from_lake_number, &
                   lake_parameters%lake_number)
    call lake_model_prognostics%set_forest%sets%reset_iterator()
    do while (.not. lake_model_prognostics%set_forest%sets%iterate_forward())
      x => lake_model_prognostics%set_forest%sets%get_value_at_iterator_position()
      root => find_root(x)
      if (root%get_label() == lake_parameters%lake_number) then
        lake_model_prognostics%lakes(x%get_label())%lake_pointer%redirect_target = -1
      end if
    end do
    call initialise_filling_lake(lake,.true.)
end subroutine change_subsumed_to_filling_lake

subroutine drain_excess_water(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  type(lakeprognostics), pointer :: other_lake
  type(rooted_tree), pointer :: x
  type(rooted_tree), pointer :: root
  real(dp) :: total_lake_volume
  real(dp) :: excess_water
  integer :: other_lake_number
  real(dp) :: flow
  integer :: lake_number_root_label
    lake_parameters => lake%parameters
    lake_model_parameters => lake%lake_model_parameters
    lake_model_prognostics => lake%lake_model_prognostics
    if (lake%excess_water > 0.0_dp) then
      if (lake%current_redirect%use_local_redirect) then
        other_lake_number = lake%current_redirect%local_redirect_target_lake_number
        other_lake => &
          lake_model_prognostics%lakes(other_lake_number)%lake_pointer
        excess_water = lake%excess_water
        lake%excess_water = 0.0_dp
        call add_water(other_lake,excess_water,.false.)
      else
        total_lake_volume = 0.0_dp
        lake_number_root_label = lake_model_prognostics%set_forest%&
          &find_root_from_label(lake_parameters%lake_number)
        call lake_model_prognostics%set_forest%sets%reset_iterator()
        do while (.not. lake_model_prognostics%set_forest%sets%iterate_forward())
          x => lake_model_prognostics%set_forest%sets%get_value_at_iterator_position()
          root => find_root(x)
          if (root%get_label() == lake_number_root_label) then
            total_lake_volume = total_lake_volume + &
              get_lake_volume(lake_model_prognostics%lakes(x%get_label())%lake_pointer)
          end if
        end do
        flow = (total_lake_volume)/ &
               (lake_model_parameters%lake_retention_constant + 1.0_dp)
        flow = min(flow,lake%excess_water)
        lake_model_prognostics%water_to_hd( &
          lake%current_redirect%_COORDS_ARG_non_local_redirect_target_) = &
             lake_model_prognostics%water_to_hd( &
                lake%current_redirect%_COORDS_ARG_non_local_redirect_target_)+flow
        lake%excess_water = lake%excess_water - flow
      end if
    end if
end subroutine drain_excess_water

subroutine release_negative_water(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
    lake_parameters => lake%parameters
    lake_model_prognostics => lake%lake_model_prognostics
    lake_model_prognostics%lake_water_from_ocean( &
              lake_parameters%_COORDS_ARG_center_cell_coarse_coords_) = &
      lake_model_prognostics%lake_water_from_ocean(lake_parameters%_COORDS_ARG_center_cell_coarse_coords_) - &
        lake%lake_volume
    lake%lake_volume = 0.0_dp
end subroutine release_negative_water

function get_lake_volume(lake) result(lake_volume)
  type(lakeprognostics), pointer, intent(in) :: lake
  real(dp) :: lake_volume
    if (lake%lake_type == filling_lake_type) then
      lake_volume = lake%lake_volume + lake%unprocessed_water
    else if (lake%lake_type == overflowing_lake_type) then
      lake_volume = lake%parameters%filling_order(lake%parameters%&
                      &number_of_cells_when_filled)%fill_threshold + &
                    lake%excess_water + lake%unprocessed_water
    else if (lake%lake_type == subsumed_lake_type) then
      lake_volume = lake%parameters%filling_order(lake%parameters%&
                      &number_of_cells_when_filled)%fill_threshold + &
                    lake%unprocessed_water
    else
      write(*,*) "Unrecognised lake type"
      stop
    end if
end function get_lake_volume

subroutine calculate_effective_lake_volume_per_cell(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
  type(lakeparameters), pointer :: lake_parameters
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  type(lakeprognostics), pointer :: other_lake
  type(rooted_tree), pointer :: x
  type(rooted_tree), pointer :: root
  real(dp) :: total_lake_volume
  real(dp) :: effective_volume_per_cell
  _DEF_COORDS_surface_model_coords_
  _DEF_COORDS_coords_
  integer :: total_number_of_flooded_cells
  integer :: lake_number_root_label
  integer :: number_of_filled_cells
  integer :: i
    if (lake%lake_type == filling_lake_type .and. &
        lake%lake_type == overflowing_lake_type) then
      lake_parameters => lake%parameters
      lake_model_parameters => lake%lake_model_parameters
      lake_model_prognostics => lake%lake_model_prognostics
      total_number_of_flooded_cells = 0
      total_lake_volume = 0.0_dp
      lake_number_root_label = lake_model_prognostics%set_forest%&
        &find_root_from_label(lake_parameters%lake_number)
      call lake_model_prognostics%set_forest%sets%reset_iterator()
      do while (.not. lake_model_prognostics%set_forest%sets%iterate_forward())
        x => lake_model_prognostics%set_forest%sets%get_value_at_iterator_position()
        root => find_root(x)
        if (root%get_label() == lake_number_root_label) then
          other_lake = lake_model_prognostics%lakes(x%get_label())%lake_pointer
          total_lake_volume = total_lake_volume + &
            get_lake_volume(other_lake)
          if (other_lake%lake_type == filling_lake_type) then
            total_number_of_flooded_cells = total_number_of_flooded_cells + &
              other_lake%current_filling_cell_index
          else
            total_number_of_flooded_cells = total_number_of_flooded_cells + &
              other_lake%parameters%number_of_cells_when_filled
          end if
        end if
      end do
      effective_volume_per_cell = &
        total_lake_volume / total_number_of_flooded_cells
      call lake_model_prognostics%set_forest%sets%reset_iterator()
      do while (.not. lake_model_prognostics%set_forest%sets%iterate_forward())
        x => lake_model_prognostics%set_forest%sets%get_value_at_iterator_position()
        root => find_root(x)
        if (root%get_label() == lake_number_root_label) then
          other_lake = lake_model_prognostics%lakes(x%get_label())%lake_pointer
          if (other_lake%lake_type == filling_lake_type) then
            number_of_filled_cells = other_lake%current_filling_cell_index
          else if (other_lake%lake_type == overflowing_lake_type .or. &
                   other_lake%lake_type == subsumed_lake_type) then
            number_of_filled_cells = other_lake%parameters%number_of_cells_when_filled
          else
            write(*,*) "Unrecognised lake type"
            stop
          end if
          do i = 1,other_lake%parameters%number_of_cells_when_filled
            _ASSIGN_COORDS_coords_ = other_lake%parameters%filling_order(i)%_COORDS_coords_
            _GET_COORDS_ _COORDS_surface_model_coords_ _FROM_ lake_model_parameters%_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _COORDS_coords_
            lake_model_prognostics%&
              &effective_volume_per_cell_on_surface_grid(_COORDS_ARG_surface_model_coords_) = &
                lake_model_prognostics%&
                &effective_volume_per_cell_on_surface_grid(_COORDS_ARG_surface_model_coords_) + &
                effective_volume_per_cell
          end do
        end if
      end do
    else if (lake%lake_type == subsumed_lake_type) then
      return
    else
      write(*,*) "Unrecognised lake type"
      stop
    end if
end subroutine calculate_effective_lake_volume_per_cell

subroutine show(lake)
  type(lakeprognostics), pointer, intent(inout) :: lake
  type(lakeparameters), pointer :: lake_parameters
    lake_parameters => lake%parameters
    write(*,*) "-----------------------------------------------"
    write(*,*) "Lake number: ", lake_parameters%lake_number
    write(*,*) "Center cell: ", lake_parameters%_COORDS_ARG_center_coords_
    write(*,*) "Unprocessed water: ", lake%unprocessed_water
    write(*,*) "Active Lake: ", lake%active_lake
    write(*,*) "Center cell coarse coords: ", lake_parameters%_COORDS_ARG_center_cell_coarse_coords_
    if (lake%lake_type == overflowing_lake_type) then
      write(*,*) "O"
      write(*,*) "Excess water: ", lake%excess_water
      write(*,*) "Current Redirect: ", lake%current_redirect
    else if (lake%lake_type == filling_lake_type) then
      write(*,*) "F"
      write(*,*) "Lake volume: ", lake%lake_volume
      write(*,*) "Current cell to fill: ", lake%_COORDS_ARG_current_cell_to_fill_
      write(*,*) "Current height type: ", lake%current_height_type
      write(*,*) "Current filling cell index: ", lake%current_filling_cell_index
      write(*,*) "Next cell volume threshold: ", lake%next_cell_volume_threshold
      write(*,*) "Previous cell volume threshold: ", lake%previous_cell_volume_threshold
    else if (lake%lake_type == subsumed_lake_type) then
      write(*,*) "S"
      write(*,*) "Redirect target : ", lake%redirect_target
    end if
end subroutine show

! !##############################################################################
! !##############################################################################
! !            Section 5: Subprograms on the full ensemble of lakes
! !##############################################################################
! !##############################################################################

subroutine create_lakes(lake_model_parameters, &
                        lake_model_prognostics, &
                        lake_parameters_as_array)
  type(lakemodelparameters), intent(inout) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), dimension(:), allocatable, intent(in) :: lake_parameters_as_array
  type(lakeparameterspointer), dimension(:), allocatable :: lake_parameters_array
  type(integerlist), pointer, dimension(:) :: basins_temp
  integer, dimension(:), allocatable :: basins_in_coarse_cell
  integer, dimension(:), allocatable :: basins_in_coarse_cell_temp
  _DEF_INDICES_LIST_cells_with_lakes_temp_INDEX_NAME_
  type(lakeparameters) :: lake_parameters
  type(lakeprognostics), pointer :: lake
  _DEF_COORDS_surface_model_coords_
  _DEF_COORDS_fine_coords_
  _DEF_SCALE_FACTORS_scale_factor_
  _DEF_LOOP_INDEX_HD_
  integer :: basin_number
  integer :: cells_with_lakes_count
  integer :: number_of_basins_in_coarse_cell
  logical :: contains_lake
  logical :: basins_found
  integer :: i
    ! lake_parameters_array = &
    !   get_lake_parameters_from_array(lake_parameters_as_array, &
    !                                  lake_model_parameters%_NPOINTS_LAKE_, &
    !                                  lake_model_parameters%_NPOINTS_HD_)
    do i = 1,size(lake_parameters_array)
      lake_parameters = lake_parameters_array(i)%lake_parameters_pointer
      if (i /= lake_parameters%lake_number) then
        write(*,*) "Lake number doesn't match position when creating lakes"
        stop
      end if
      call add_set(lake_model_prognostics%set_forest,lake_parameters%lake_number)
      lake = lakeprognostics(lake_parameters, &
                             lake_model_parameters, &
                             lake_model_prognostics)
      lake_model_prognostics%lakes(i) = lakepointer(lake)
      if (lake_parameters%is_leaf .and. (lake%current_height_type == flood_height &
                                  .or. size(lake_parameters%filling_order) == 1)) then
        _GET_COORDS_ _COORDS_surface_model_coords_ _FROM_ lake_model_parameters%_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ lake%_COORDS_current_cell_to_fill_
        lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) = &
          lake_model_prognostics%lake_cell_count(_COORDS_ARG_surface_model_coords_) + 1
        lake_model_prognostics%lake_numbers(lake%_COORDS_ARG_current_cell_to_fill_) = &
          lake%parameters%lake_number
      end if
    end do
    do i=1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      lake_model_parameters%lake_centers(lake%parameters%_COORDS_ARG_center_coords_) = .true.
    end do
    allocate(basins_in_coarse_cell_temp((lake_model_parameters%_NPOINTS_TOTAL_LAKE_)&
                                        /(lake_model_parameters%_NPOINTS_TOTAL_HD_)))
    allocate(lake_model_parameters%basin_numbers(lake_model_parameters%_NPOINTS_HD_))
    allocate(basins_temp(lake_model_parameters%_NPOINTS_TOTAL_HD_))
    basin_number = 1
    cells_with_lakes_count = 0
    _CALCULATE_SCALE_FACTORS_scale_factor_ _lake_model_parameters%_NPOINTS_LAKE_ _lake_model_parameters%_NPOINTS_HD_
    _LOOP_OVER_HD_GRID_ _COORDS_HD_ _lake_model_parameters%_
      number_of_basins_in_coarse_cell = 0
      contains_lake = .false.
      _FOR_FINE_CELLS_IN_COARSE_CELL_ _COORDS_fine_coords_ _COORDS_HD_ _SCALE_FACTORS_scale_factor_
        if (lake_model_parameters%lake_centers(_COORDS_ARG_fine_coords_)) then
          contains_lake = .true.
          do i=1,size(lake_model_prognostics%lakes)
            lake => lake_model_prognostics%lakes(i)%lake_pointer
            if (_EQUALS_lake%parameters%_COORDS_center_coords_ == _COORDS_fine_coords_) then
              number_of_basins_in_coarse_cell = &
                number_of_basins_in_coarse_cell + 1
              basins_in_coarse_cell_temp(number_of_basins_in_coarse_cell) = &
                lake%parameters%lake_number
            end if
          end do
        end if
      _END_FOR_FINE_CELLS_IN_COARSE_CELL_
      if (contains_lake) then
        _ASSIGN_INDICES_LIST_cells_with_lakes_temp_INDEX_NAME(basin_number)_ = _COORDS_HD_
        cells_with_lakes_count = cells_with_lakes_count + 1
      end if
      if(number_of_basins_in_coarse_cell > 0) then
        allocate(basins_in_coarse_cell(number_of_basins_in_coarse_cell))
        do i=1,number_of_basins_in_coarse_cell
          basins_in_coarse_cell(i) = basins_in_coarse_cell_temp(i)
        end do
        basin_number = basin_number + 1
        basins_temp(basin_number) = &
          integerlist(basins_in_coarse_cell)
        lake_model_parameters%basin_numbers(_COORDS_HD_) = basin_number
      end if
    _LOOP_OVER_HD_GRID_END_
    do i=1,cells_with_lakes_count
      allocate(lake_model_parameters%_INDICES_LIST_cells_with_lakes_INDEX_NAME(cells_with_lakes_count)_)
      _ASSIGN_lake_model_parameters%_INDICES_LIST_cells_with_lakes_INDEX_NAME(i)_ = &
        _INDICES_LIST_cells_with_lakes_temp_INDEX_NAME(i)_
    end do
    allocate(lake_model_parameters%basins(basin_number))
    do i=1,basin_number
      lake_model_parameters%basins(i) = basins_temp(i)
    end do
    deallocate(basins_temp)
    deallocate(basins_in_coarse_cell_temp)
end subroutine create_lakes

subroutine setup_lakes(lake_model_parameters,lake_model_prognostics,&
                       initial_water_to_lake_centers)
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), dimension(_DIMS_), intent(in) :: initial_water_to_lake_centers
  type(lakeprognostics), pointer :: lake
  real(dp) :: initial_water_to_lake_center
  integer :: lake_index
  _DEF_LOOP_INDEX_LAKE_
    _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_  _lake_model_parameters%_
      if (lake_model_parameters%lake_centers(_COORDS_LAKE_)) then
        initial_water_to_lake_center = &
          initial_water_to_lake_centers(_COORDS_LAKE_)
        if (initial_water_to_lake_center > 0.0_dp) then
          lake_index = lake_model_prognostics%lake_numbers(_COORDS_LAKE_)
          lake => lake_model_prognostics%lakes(lake_index)%lake_pointer
          call add_water(lake,initial_water_to_lake_center,.false.)
        end if
      end if
    _LOOP_OVER_LAKE_GRID_END_
end subroutine setup_lakes

subroutine distribute_spillover(lake_model_parameters,lake_model_prognostics, &
                              initial_spillover_to_rivers)
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), dimension(_DIMS_), intent(in) :: initial_spillover_to_rivers
  real(dp) :: initial_spillover
  _DEF_LOOP_INDEX_HD_
    _LOOP_OVER_HD_GRID_ _COORDS_HD_  _lake_model_parameters%_
      initial_spillover = initial_spillover_to_rivers(_COORDS_HD_)
      if (initial_spillover > 0.0_dp) then
        lake_model_prognostics%water_to_hd(_COORDS_HD_) = &
          lake_model_prognostics%water_to_hd(_COORDS_HD_) + initial_spillover
      end if
    _LOOP_OVER_HD_GRID_END_
end subroutine distribute_spillover

subroutine run_lakes(lake_model_parameters,lake_model_prognostics)
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  type(lakeprognostics), pointer :: lake
  type(coordslist), pointer :: coords_list
  integer, dimension(:), pointer :: lakes_in_cell
  real(dp) :: evaporation_per_lake_cell
  real(dp) :: share_to_each_lake
  real(dp) :: inflow_minus_evaporation
  real(dp) :: evaporation
  integer :: active_lakes_in_cell_count
  integer :: lake_cell_count
  integer :: cell_count_check
  integer :: map_index
  integer :: target_lake_index
  integer :: lake_index
  integer :: basin_number
  integer :: i,j
  _DEF_COORDS_coords_
  _DEF_COORDS_fine_coords_
  _DEF_LOOP_INDEX_SURFACE_
    lake_model_prognostics%water_to_hd(_DIMS_) = 0.0_dp
    lake_model_prognostics%lake_water_from_ocean(_DIMS_) = 0.0_dp
    lake_model_prognostics%evaporation_from_lakes(:) = 0.0_dp
    lake_model_prognostics%new_effective_volume_per_cell_on_surface_grid(_DIMS_) = &
      lake_model_prognostics%effective_lake_height_on_surface_grid_to_lakes(_DIMS_)* &
      lake_model_parameters%cell_areas_on_surface_model_grid(_DIMS_)
    lake_model_prognostics%evaporation_on_surface_grid(_DIMS_) = &
      lake_model_prognostics%effective_volume_per_cell_on_surface_grid(_DIMS_) - &
      lake_model_prognostics%new_effective_volume_per_cell_on_surface_grid(_DIMS_)
    lake_model_prognostics%effective_volume_per_cell_on_surface_grid(_DIMS_) = 0.0_dp
    _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_ _lake_model_parameters%_
      lake_cell_count = lake_model_prognostics%lake_cell_count(_COORDS_SURFACE_)
      if (lake_model_prognostics%evaporation_on_surface_grid(_COORDS_SURFACE_) /= 0.0_dp &
          .and. lake_cell_count > 0) then
        cell_count_check = 0
        map_index = lake_model_parameters%surface_cell_to_fine_cell_map_numbers(_COORDS_SURFACE_)
        evaporation_per_lake_cell = &
          lake_model_prognostics%evaporation_on_surface_grid(_COORDS_SURFACE_)/lake_cell_count
        if (map_index /= 0) then
          coords_list => lake_model_parameters%surface_cell_to_fine_cell_maps(map_index)
          do i=1,size(coords_list%_INDICES_LIST_INDEX_NAME_coords_FIRST_DIM_)
            _GET_COORDS_ _COORDS_fine_coords_ _FROM_ coords_list%_INDICES_LIST_INDEX_NAME_coords_ i
            target_lake_index = lake_model_prognostics%lake_numbers(_COORDS_ARG_fine_coords_)
            if (target_lake_index /= 0) then
              lake => lake_model_prognostics%lakes(target_lake_index)%lake_pointer
              lake_model_prognostics%evaporation_from_lakes(target_lake_index) = &
                lake_model_prognostics%evaporation_from_lakes(target_lake_index) + evaporation_per_lake_cell
              cell_count_check = cell_count_check + 1
            end if
          end do
        end if
        if (cell_count_check /= lake_cell_count) then
          write(*,*) "Inconsistent cell count when assigning evaporation"
          stop
        end if
      end if
    _LOOP_OVER_SURFACE_GRID_END_
    lake_model_prognostics%evaporation_applied(:) = .false.
    do i=1,size(lake_model_parameters%_INDICES_LIST_cells_with_lakes_INDEX_NAME_FIRST_DIM_)
      if (lake_model_prognostics%water_to_lakes(_COORDS_ARG_coords_) > 0.0_dp) then
        basin_number = lake_model_parameters%basin_numbers(_COORDS_ARG_coords_)
        lakes_in_cell => &
          lake_model_parameters%basins(basin_number)%list
        active_lakes_in_cell_count = 0
        do j=1,size(lakes_in_cell)
          lake_index = lakes_in_cell(j)
          lake => lake_model_prognostics%lakes(lake_index)%lake_pointer
          if (lake%active_lake) then
            active_lakes_in_cell_count = active_lakes_in_cell_count + 1
          end if
        end do
        share_to_each_lake = lake_model_prognostics%water_to_lakes(_COORDS_ARG_coords_)/ &
                             active_lakes_in_cell_count
        do j=1,size(lakes_in_cell)
          lake_index = lakes_in_cell(j)
          lake => lake_model_prognostics%lakes(lake_index)%lake_pointer
          if (lake%active_lake) then
            if (.not. lake_model_prognostics%evaporation_applied(lake_index)) then
              inflow_minus_evaporation = share_to_each_lake - &
                                         lake_model_prognostics%evaporation_from_lakes(lake_index)
              if (inflow_minus_evaporation >= 0.0_dp) then
                call add_water(lake,inflow_minus_evaporation,.true.)
              else
                call remove_water(lake,-1.0_dp*inflow_minus_evaporation,.true.)
              end if
              lake_model_prognostics%evaporation_applied(lake_index) = .true.
            else
              call add_water(lake,share_to_each_lake,.true.)
            end if
          end if
        end do
      else if (lake_model_prognostics%water_to_lakes(_COORDS_ARG_coords_) < 0.0_dp) then
        lakes_in_cell = &
          lake_model_parameters%&
            &basins(lake_model_parameters%basin_numbers(_COORDS_ARG_coords_))%list
        active_lakes_in_cell_count = 0
        do j=1,size(lakes_in_cell)
          lake_index = lakes_in_cell(j)
          lake => lake_model_prognostics%lakes(lake_index)%lake_pointer
          if (lake%active_lake) then
            active_lakes_in_cell_count = active_lakes_in_cell_count + 1
          end if
        end do
        share_to_each_lake = &
          -1.0_dp*lake_model_prognostics%water_to_lakes(_COORDS_ARG_coords_)/&
          active_lakes_in_cell_count
        do j=1,size(lakes_in_cell)
          lake_index = lakes_in_cell(j)
          lake => lake_model_prognostics%lakes(lake_index)%lake_pointer
          if (lake%active_lake) then
            call remove_water(lake,share_to_each_lake,.true.)
          end if
        end do
      end if
    end do
    do i = 1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      lake_index = lake%parameters%lake_number
      if (.not. lake_model_prognostics%evaporation_applied(lake_index)) then
        evaporation = lake_model_prognostics%evaporation_from_lakes(lake_index)
        if (evaporation > 0.0) then
          call remove_water(lake,evaporation,.true.)
        else if (evaporation < 0.0) then
          call add_water(lake,-1.0_dp*evaporation,.true.)
        end if
        lake_model_prognostics%evaporation_applied(lake_index) = .true.
      end if
    end do
    do i=1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      if (lake%unprocessed_water /= 0.0_dp) then
        call process_water(lake)
      end if
    end do
    do i=1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      if (lake%lake_type == filling_lake_type) then
        if (lake%lake_volume < 0.0_dp) then
          call release_negative_water(lake)
        end if
      end if
    end do
    do i=1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      if (lake%lake_type == overflowing_lake_type) then
        call drain_excess_water(lake)
      end if
    end do
    do i=1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      call calculate_effective_lake_volume_per_cell(lake)
    end do
end subroutine run_lakes

subroutine set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters, &
                                                              lake_model_prognostics, &
                                                              effective_lake_height_on_surface_grid)
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), dimension(_DIMS_), intent(in) :: effective_lake_height_on_surface_grid
  _DEF_LOOP_INDEX_SURFACE_
    _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_  _lake_model_parameters%_
      lake_model_prognostics%effective_lake_height_on_surface_grid_to_lakes(_COORDS_SURFACE_) = &
        effective_lake_height_on_surface_grid(_COORDS_SURFACE_)
    _LOOP_OVER_SURFACE_GRID_END_
end subroutine set_effective_lake_height_on_surface_grid_to_lakes

subroutine calculate_lake_fraction_on_surface_grid(lake_model_parameters, &
                                                   lake_model_prognostics, &
                                                   lake_fraction_on_surface_grid)
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), dimension(:,:), intent(inout) :: lake_fraction_on_surface_grid
    lake_fraction_on_surface_grid(_DIMS_) = real(lake_model_prognostics%lake_cell_count(_DIMS_))&
                                         /real(lake_model_parameters%number_fine_grid_cells(_DIMS_))
end subroutine calculate_lake_fraction_on_surface_grid

subroutine calculate_effective_lake_height_on_surface_grid(&
    lake_model_parameters, &
    lake_model_prognostics, &
    effective_lake_height_on_surface_grid_from_lakes)
  type(lakemodelparameters), intent(in) :: lake_model_parameters
  type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
  real(dp), dimension(:,:), intent(inout) :: effective_lake_height_on_surface_grid_from_lakes
    effective_lake_height_on_surface_grid_from_lakes(_DIMS_) = &
      lake_model_prognostics%effective_volume_per_cell_on_surface_grid(_DIMS_) / &
      lake_model_parameters%cell_areas_on_surface_model_grid(_DIMS_)
end subroutine calculate_effective_lake_height_on_surface_grid

! subroutine write_lake_numbers(lake_model_parameters,lake_model_prognostics, &
!                               timestep)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     call write_lake_numbers_field(lake_model_parameters%lake_model_grid, &
!                                   lake_model_prognostics%lake_numbers, &
!                                   timestep=timestep)
! end subroutine write_lake_numbers

! subroutine write_lake_volumes(lake_model_parameters,lake_model_prognostics)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   type(lakeprognostics), pointer :: lake
!     lake_volumes(_DIMS_) = 0.0_dp
!     do i=1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       lake_center_cell::CartesianIndex = lake%parameters%center_cell
!       lake_volumes(lake_center_cell) = get_lake_volume(lake)
!     end do
!     call write_lake_volumes_field(lake_model_parameters%lake_model_grid,lake_volumes)
! end subroutine write_lake_volumes

! subroutine write_diagnostic_lake_volumes(lake_model_parameters,lake_model_prognostics, &
!                                          timestep)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     diagnostic_lake_volumes(_DIMS_) = &
!       calculate_diagnostic_lake_volumes_field(lake_model_parameters, &
!                                               lake_model_prognostics)
!     write_diagnostic_lake_volumes_field(lake_model_parameters%lake_model_grid, &
!                                         diagnostic_lake_volumes, &
!                                         timestep)
! end subroutine write_diagnostic_lake_volumes

function calculate_diagnostic_lake_volumes_field(lake_model_parameters, &
                                                 lake_model_prognostics) &
    result(diagnostic_lake_volumes)
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volumes
  real(dp), dimension(:), allocatable :: lake_volumes_by_lake_number
  type(lakeprognostics), pointer :: lake
  type(rooted_tree), pointer :: x
  type(rooted_tree), pointer :: root
  real(dp) :: total_lake_volume
  integer :: lake_number
  integer :: lake_number_root_label
  integer :: i
  _DEF_LOOP_INDEX_LAKE_
    allocate(diagnostic_lake_volumes(lake_model_parameters%_NPOINTS_LAKE_))
    diagnostic_lake_volumes(_DIMS_) = 0.0_dp
    allocate(lake_volumes_by_lake_number(size(lake_model_prognostics%lakes)))
    lake_volumes_by_lake_number(:) = 0.0_dp
    do i = 1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      if (lake%active_lake) then
        total_lake_volume = 0.0_dp
        lake_number_root_label = lake_model_prognostics%set_forest%&
          &find_root_from_label(lake%parameters%lake_number)
        call lake_model_prognostics%set_forest%sets%reset_iterator()
        do while (.not. lake_model_prognostics%set_forest%sets%iterate_forward())
          x => lake_model_prognostics%set_forest%sets%get_value_at_iterator_position()
          root => find_root(x)
          if (root%get_label() == lake_number_root_label) then
            total_lake_volume = total_lake_volume + &
              get_lake_volume(lake_model_prognostics%lakes(x%get_label())%lake_pointer)
          end if
          lake_volumes_by_lake_number(i) = total_lake_volume
        end do
      end if
    end do
    _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_ _lake_model_parameters%_
      lake_number = lake_model_prognostics%lake_numbers(_COORDS_LAKE_)
      if (lake_number > 0) then
        diagnostic_lake_volumes(_COORDS_LAKE_) = lake_volumes_by_lake_number(lake_number)
      end if
    _LOOP_OVER_LAKE_GRID_END_
    deallocate(lake_volumes_by_lake_number)
end function calculate_diagnostic_lake_volumes_field

subroutine set_lake_evaporation_for_testing(lake_model_parameters,lake_model_prognostics, &
                                lake_evaporation)
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  real(dp), dimension(_DIMS_), intent(in) :: lake_evaporation
  real(dp), dimension(_DIMS_), allocatable :: effective_lake_height_on_surface_grid
  real(dp), dimension(_DIMS_), allocatable :: old_effective_lake_height_on_surface_grid
    call calculate_effective_lake_height_on_surface_grid( &
      lake_model_parameters,lake_model_prognostics, &
      old_effective_lake_height_on_surface_grid)
    effective_lake_height_on_surface_grid(_DIMS_) = &
      old_effective_lake_height_on_surface_grid(_DIMS_) - &
        (lake_evaporation(_DIMS_)/ &
         lake_model_parameters%cell_areas_on_surface_model_grid(_DIMS_))
    call set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters, &
                                                            lake_model_prognostics, &
                                                            effective_lake_height_on_surface_grid)
end subroutine set_lake_evaporation_for_testing

subroutine set_lake_evaporation(lake_model_parameters,lake_model_prognostics, &
                                height_of_water_evaporated)
  type(lakemodelparameters), pointer, intent(in) :: lake_model_parameters
  type(lakemodelprognostics), pointer, intent(inout) :: lake_model_prognostics
  real(dp), dimension(_DIMS_), intent(in) :: height_of_water_evaporated
  real(dp), dimension(_DIMS_), allocatable :: effective_lake_height_on_surface_grid
  real(dp), dimension(_DIMS_), allocatable :: old_effective_lake_height_on_surface_grid
  real(dp) :: working_effective_lake_height_on_surface_grid
  _DEF_LOOP_INDEX_SURFACE_
    call calculate_effective_lake_height_on_surface_grid(&
      lake_model_parameters, lake_model_prognostics, &
      old_effective_lake_height_on_surface_grid)
    effective_lake_height_on_surface_grid(_DIMS_) = &
      old_effective_lake_height_on_surface_grid
    _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_ _lake_model_parameters%_
      if (old_effective_lake_height_on_surface_grid(_COORDS_SURFACE_) > 0.0_dp) then
        working_effective_lake_height_on_surface_grid = &
          old_effective_lake_height_on_surface_grid(_COORDS_SURFACE_) - &
          height_of_water_evaporated(_COORDS_SURFACE_)
        if (working_effective_lake_height_on_surface_grid > 0.0_dp) then
          effective_lake_height_on_surface_grid(_COORDS_SURFACE_) = &
            working_effective_lake_height_on_surface_grid
        else
          effective_lake_height_on_surface_grid(_COORDS_SURFACE_) = 0.0_dp
        end if
      end if
    _LOOP_OVER_SURFACE_GRID_END_
    call set_effective_lake_height_on_surface_grid_to_lakes(&
              lake_model_parameters,lake_model_prognostics, &
              effective_lake_height_on_surface_grid)
end subroutine set_lake_evaporation

subroutine check_water_budget(lake_model_prognostics, &
                              total_initial_water_volume)
  type(lakemodelparameters), pointer :: lake_model_parameters
  type(lakemodelprognostics), pointer :: lake_model_prognostics
  real(dp), intent(in), optional :: total_initial_water_volume
  type(lakeprognostics), pointer :: lake
  real(dp) :: total_initial_water_volume_local
  real(dp) :: new_total_lake_volume
  real(dp) :: change_in_total_lake_volume
  real(dp) :: total_water_to_lakes
  real(dp) :: total_inflow_minus_outflow
  real(dp) :: difference
  real(dp) :: tolerance
  integer :: i
    if (present(total_initial_water_volume)) then
      total_initial_water_volume_local = total_initial_water_volume
    else
      total_initial_water_volume_local = 0
    end if
    new_total_lake_volume = 0.0_dp
    do i = 1,size(lake_model_prognostics%lakes)
      lake => lake_model_prognostics%lakes(i)%lake_pointer
      new_total_lake_volume = new_total_lake_volume + get_lake_volume(lake)
    end do
    change_in_total_lake_volume = new_total_lake_volume - &
                                  lake_model_prognostics%total_lake_volume - &
                                  total_initial_water_volume_local
    total_water_to_lakes = sum(lake_model_prognostics%water_to_lakes)
    total_inflow_minus_outflow = total_water_to_lakes + &
                                 sum(lake_model_prognostics%lake_water_from_ocean) - &
                                 sum(lake_model_prognostics%water_to_hd) - &
                                 sum(lake_model_prognostics%evaporation_from_lakes)
    difference = change_in_total_lake_volume - total_inflow_minus_outflow
    tolerance = 1.0e-13_dp*max(new_total_lake_volume, total_water_to_lakes)
    if (abs(difference) >= tolerance) then
      write(*,*) "*** Lake Water Budget ***"
      write(*,*) "Total lake volume: ",new_total_lake_volume
      write(*,*) "Total inflow - outflow: ", total_inflow_minus_outflow
      write(*,*) "Change in lake volume:", change_in_total_lake_volume
      write(*,*) "Difference: ", difference
      write(*,*) "Total water to lakes: ", total_water_to_lakes
      write(*,*) "Total water from ocean: ", sum(lake_model_prognostics%lake_water_from_ocean)
      write(*,*) "Total water to HD model from lakes: ", sum(lake_model_prognostics%water_to_hd)
      write(*,*) "Old total lake volume: ", lake_model_prognostics%total_lake_volume
    end if
    lake_model_prognostics%total_lake_volume = new_total_lake_volume
end subroutine check_water_budget

end module l2_lake_model_mod
