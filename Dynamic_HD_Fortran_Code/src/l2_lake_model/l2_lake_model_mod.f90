module l2_lake_model_mod

use latlon_lake_model_tree_mod

! TO DO
! coords
! loops
! dict --> sort using selection sort then
!          navigate by division if large
! debug printout etc
! basins
! finish constructors --> cell,redirect
! lake pointers func
! lake pointers alloc
! clean up excess brackets in if statements
! remove doubling up of vars in constructor

! use latlon_lake_model_tree_mod
! #ifdef USE_LOGGING
!   use latlon_lake_logger_mod
! #endif

integer,parameter :: dp = selected_real_kind(12)

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
  INDICES_coords
end type coordslist

interface coordslist
  procedure :: cooordslistconstructor
end interface

type :: integerlist
  integer, pointer, dimension(:) :: list
end type

interface integerlist
  procedure :: integerlistconstructor
end interface

type :: lakemodelparameters
  _DEF_INDICES_corresponding_surface_cell_INDEX_NAME_index
  _DEF_NPOINTS_LAKE_
  _DEF_NPOINTS_HD_
  _DEF_NPOINTS_SURFACE_
  real(dp) :: lake_retention_constant
  real(dp) :: minimum_lake_volume_threshold
  integer :: number_of_lakes
  type(integerlist), pointer, dimension(_DIMS_) :: basins
  integer, pointer, dimension(_DIMS_) :: basin_numbers
  _DEF_INDICES_cells_with_lakes_INDEX_NAME
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
  COORDS_coords
  integer :: height_type
  real(dp) :: fill_threshold
  real(dp) :: height
end type cell

interface cell
  procedure :: cellconstructors
end interface cell

type :: redirect
  logical :: use_local_redirect
  integer :: local_redirect_target_lake_number
  DEF_COORDS_non_local_redirect_target
end type redirect

interface redirect
  procedure :: redirectconstructor
end interface redirect

type :: lakeparameters
  COORDS_center_coords
  COORDS_center_cell_coarse_coords
  integer :: lake_number
  logical :: is_primary
  logical :: is_leaf
  integer :: primary_lake
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
  COORDS_current_cell_to_fill
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

contains

!##############################################################################
!##############################################################################
!              Section 2: Constructors and Initialisation Routines
!##############################################################################
!##############################################################################

! function redirectdictionaryconstructor(number_of_entries) result(constructor)
!   integer, intent(in) :: number_of_entries
!   type(redirectdictionary) :: constructor
!     allocate(constructor%keys(number_of_entries))
!     allocate(constructor%values(number_of_entries))
!     constructor%initialisation_complete = .false.
!     constructor%next_entry_index = 1
! end function redirectdictionaryconstructor

! function lakemodelparametersconstructor(
!                              INDICES_corresponding_surface_cell_INDEX_NAME_index,
!                              cell_areas_on_surface_model_grid,
!                              number_of_lakes,
!                              is_lake,
!                              NPOINTS_hd,
!                              NPOINTS_lake,
!                              NPOINTS_surface,
!                              lake_retention_constant,
!                              minimum_lake_volume_threshold) result(constructor)
!   integer, intent(in) :: number_of_lakes
!   _DEF_NPOINTS_hd IN
!   _DEF_NPOINTS_lake IN
!   _DEF_NPOINTS_surface IN
!   _DEF_INDICES_corresponding_surface_cell_INDEX_NAME_index IN
!   real(dp), pointer, dimension(_DIMS_), intent(in) :: cell_areas_on_surface_model_grid
!   real(dp), intent(in), optional :: lake_retention_constant
!   real(dp), intent(in), optional :: minimum_lake_volume_threshold
!   type(lakemodelparameters) :: constructor
!   type(integerlist), pointer, dimension(_DIMS_) :: basins
!   integer, pointer, dimension(_DIMS_) :: basin_numbers
!   _DEF_INDICES_cells_with_lakes_INDEX_NAME
!   integer, pointer, dimension(_DIMS_) :: number_of_lake_cells_temp
!   logical, pointer, dimension(:,:) :: needs_map
!   type(coordslist), pointer, dimension(:) :: surface_cell_to_fine_cell_maps
!   integer, pointer, dimension(_DIMS_) :: surface_cell_to_fine_cell_map_numbers
!   integer :: k
!     if (.not. present(lake_retention_constant)) then
!       lake_retention_constant = 0.1_dp
!     end if
!     if (.not. present(minimum_lake_volume_threshold)) then
!       minimum_lake_volume_threshold = 0.0000001_dp
!     end if
!     allocate(this%number_fine_grid_cells(NPOINTS_surface))
!     this%number_fine_grid_cells(_DIMS_) = 0
!     allocate(needs_map(NPOINTS_surface))
!     needs_map(_DIMS_) = .false.
!     allocate(number_of_lake_cells_temp(NPOINTS_SURFACE))
!     number_of_lake_cells_temp(_DIMS_) = 0
!     _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
!       _GET_COORDS_ _SURFACE_MODEL_COORDS_ _FROM_ INDICES_corresponding_surface_cell_INDEX_NAME_index _COORDS_LAKE_
!       number_fine_grid_cells(
!            _SURFACE_MODEL_COORDS_) = &
!            number_fine_grid_cells(_SURFACE_MODEL_COORDS_) + 1
!       if (is_lake(_COORDS_LAKE_)) then
!           needs_map(lat_surface_model,lon_surface_model) = .true.
!           number_of_lake_cells_temp(lat_surface_model,lon_surface_model) = &
!             number_of_lake_cells_temp(lat_surface_model,lon_surface_model) + 1
!       end if
!     _LOOP_OVER_LAKE_GRID_END_
!     allocate(this%surface_cell_to_fine_cell_map_numbers(NPOINTS_surface))
!     this%surface_cell_to_fine_cell_map_numbers(_DIMS_) = 0
!     map_number = 0
!     _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
!       if (needs_map(_COORDS_SURFACE)) then
!         map_number = map_number + 1
!       end if
!     _LOOP_OVER_SURFACE_GRID_END_
!     allocate(this%surface_cell_to_fine_cell_maps(map_number))
!     map_number = 0
!     _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
!       if (needs_map(_COORDS_SURFACE_)) then
!         map_number = map_number + 1
!         number_of_lake_cells = number_of_lake_cells_temp(_COORDS_SURFACE_)
!         allocate(INDICES_surface_cell_to_fine_cell_map_INDEX_NAME_PLURAL_temp(number_of_lake_cells))
!         INDICES_surface_cell_to_fine_cell_map_INDEX_NAME_PLURAL_temp(:) = -1
!         ASSIGN_this%surface_cell_to_fine_cell_maps(map_number) = &
!           coordslist(INDICES_surface_cell_to_fine_cell_map_INDEX_NAME_PLURAL_temp)
!         ASSIGN_this%surface_cell_to_fine_cell_map_numbers(_COORDS_SURFACE_) = map_number
!       end if
!     _LOOP_OVER_SURFACE_GRID_END_
!     deallocate(number_of_lake_cells_temp)
!     _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
!       if (is_lake(_COORDS_LAKE_)) then
!         _COORDS_SURFACE_ = this%INDICES_corresponding_surface_cell_INDEX_NAME_index(_COORDS_LAKE)
!         ASSIGN_surface_cell_to_fine_cell_map_number = &
!           surface_cell_to_fine_cell_map_numbers(_COORDS_SURFACE_)
!         k = 1
!         do while (this%surface_cell_to_fine_cell_maps(&
!                   surface_cell_to_fine_cell_map_number)%INDICES_coords_FIRST_DIM(k) /= -1)
!           k = k + 1
!         end do
!         ASSIGN_this%surface_cell_to_fine_cell_maps(&
!                       surface_cell_to_fine_cell_map_number)%INDICES_coords(k) = _COORDS_SURFACE
!       end if
!     _LOOP_OVER_LAKE_GRID_END_
!     deallocate(needs_map)
!     constructor%lake_retention_constant = lake_retention_constant
!     constructor%minimum_lake_volume_threshold = minimum_lake_volume_threshold
!     constructor%lake_model_grid = lake_model_grid
!     constructor%hd_model_grid = hd_model_grid
!     constructor%surface_model_grid = surface_model_grid
!     constructor%lake_model_settings = lake_model_settings
!     constructor%number_of_lakes = number_of_lakes
!     constructor%basins = basins
!     constructor%basin_numbers(_DIMS_) = 0
!     ASSIGN_constructor%cells_with_lakes = cells_with_lakes
!     constructor%cell_areas_on_surface_model_grid = cell_areas_on_surface_model_grid
!     constructor%lake_centers(_DIMS_) = .false.
!     constructor%number_fine_grid_cells(_DIMS_) = 0
!     constructor%surface_cell_to_fine_cell_maps = surface_cell_to_fine_cell_maps
!     constructor%surface_cell_to_fine_cell_map_numbers = surface_cell_to_fine_cell_map_numbers
! end function lakemodelparametersconstructor

! function lakemodelprognosticsconstructor(lake_model_parameters::LakeModelParameters) result(constructor)
!     type(lakemodelparameters), intent(in) :: lake_model_parameters
!     type(lakemodelprognostics), intent(out) :: constructor
!       allocate(constructor%lakes(lake_model_parameters%number_of_lakes))
!       constructor%lake_numbers(_DIMS_) = 0
!       constructor%lake_cell_count(_DIMS_) = 0
!       constructor%effective_volume_per_cell_on_surface_grid(_DIMS_) = 0.0_dp
!       constructor%effective_lake_height_on_surface_grid_to_lakes(_DIMS_) = 0.0_dp
!       constructor%water_to_lakes(_DIMS_)= 0.0_dp
!       constructor%water_to_hd(_DIMS_) = 0.0_dp
!       constructor%lake_water_from_ocean(_DIMS_) = 0.0_dp
!       constructor%evaporation_from_lakes = evaporation_from_lakes
!       allocate(constructor%evaporation_from_lakes(lake_model_parameters%number_of_lakes)
!       constructor%evaporation_from_lakes(:) = 0.0_dp
!       allocate(constructor%evaporation_applied(lake_model_parameters%number_of_lakes))
!       constructor%evaporation_applied(_DIMS_) = .false.
!       constructor%set_forest => rooted_tree_forest()
! end function lakemodelprognosticsconstructor

! function cellconstructor(height_type,
!                          fill_threshold,
!                          height) result(constructor)
!   integer  :: height_type
!   real(dp) :: fill_threshold
!   real(dp) :: height
!   type(constructor), pointer :: constructor
!     allocate(constructor)
!     constructor%height_type = height_type
!     constructor%fill_threshold = fill_threshold
!     constructor%height = height
! end function cellconstructor

! function redirectconstructor(use_local_redirect, &
!                          local_redirect_target_lake_number, &
!                          _COORDS_non_local_redirect_target_) &
!                         result(constructor)
!   logical, intent(in) :: use_local_redirect
!   integer, intent(in) :: local_redirect_target_lake_number
!   _DEF_COORDS_non_local_redirect_target
!   type(constructor), pointer :: constructor
!     allocate(constructor)
!     constructor%use_local_redirect = use_local_redirect
!     constructor%local_redirect_target_lake_number = &
!       local_redirect_target_lake_number
!     ASSIGN_constructor%COORDS_non_local_redirect_target = &
!       COORDS_non_local_redirect_target
! end function redirectconstructor

! function lakeparametersconstructor(lake_number,
!                                    primary_lake,
!                                    secondary_lakes,
!                                    COORDS_center_coords,
!                                    filling_order::Vector{Cell},
!                                    outflow_points::Dict{Int64,Redirect},
!                                    NPOINTS_LAKE_
!                                    NPOINTS_SURFACE_) &
!                                    result(constructor)
!   integer, intent(in) :: lake_number
!   integer, intent(in) :: primary_lake
!   integer, dimension(:), allocatable, intent(in) :: secondary_lakes
!   _DEF_COORDS_center_coords IN
!   type(cell), allocatable, dimension(:), intent(in) :: filling_order
!   type(redirectdictionary), allocatable, intent(in) :: outflow_points
!   _DEF_NPOINTS_lake IN
!   _DEF_NPOINTS_surface IN
!   type(lake_parameters), target :: constructor
!   logical :: is_primary
!   logical :: is_leaf
!     allocate(constructor)
!     ASSIGN_constructor%COORDS_center_coords = COORDS_center_coords
!     constructor%center_cell_coarse_coords = &
!       find_coarse_cell_containing_fine_cell(NPOINTS_LAKE_
!                                             NPOINTS_SURFACE_,
!                                             center_coords)
!     constructor%lake_number = lake_number
!     constructor%is_primary = (primary_lake == -1)
!     if (is_primary .and. size(outflow_points) > 1) then
!       write(*,*) "Primary lake has more than one outflow point"
!       stop
!     end if
!     constructor%is_leaf = (size(secondary_lakes) == 0)
!     constructor%primary_lake = primary_lake
!     constructor%secondary_lakes = secondary_lakes
!     constructor%filling_order = filling_order
!     constructor%outflow_points = outflow_points
! end function lake_parameters_constructor

! !Construct an empty filling lake by default
! function lakeprognosticsconstructor(lake_parameters,
!                                     lake_model_parameters,
!                                     lake_model_prognostics) result(constructor)
!   type(lakeparameters), intent(in) :: lake_parameters
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(in) :: lake_model_prognostics
!   type(lakeprognostics), pointer :: constructor
!     allocate(constructor)
!     constructor%lake_type = filling_lake_type
!     constructor%active_lake = lake_parameters%is_leaf
!     constructor%unprocessed_water = 0.0_dp
!     constructor%parameters = lake_parameters
!     constructor%lake_model_parameters = lake_model_parameters
!     constructor%lake_model_prognostics = lake_model_prognostics
!     call initialise_filling_lake(lake,.false.)
! end function lakeprognosticsconstructor

! subroutine initialise_filling_lake(lake,initialise_filled)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   logical, intent(in) :: initialise_filled
!     call set_overflowing_lake_parameter_to_null_values(lake)
!     call set_subsumed_lake_parameter_to_null_values(lake)
!     if (initialise_filled) then
!       lake%current_cell_to_fill = lake_parameters%filling_order(end)%coords
!       lake%current_height_type = lake_parameters%filling_order(end)%height_type
!       lake%current_filling_cell_index = size(lake_parameters%filling_order)
!       lake%next_cell_volume_threshold = &
!         lake_parameters%filling_order(end)%fill_threshold
!       if (size(lake_parameters%filling_order) > 1) then
!         lake%previous_cell_volume_threshold = &
!           lake_parameters%filling_order(end-1)%fill_threshold
!       else
!         lake%previous_cell_volume_threshold = 0.0_dp
!       end if
!       lake%lake_volume = lake_parameters%filling_order(end)%fill_threshold
!     else
!       lake%current_cell_to_fill = &
!         lake_parameters%filling_order(1)%coords
!       lake%current_height_type = &
!         lake_parameters%filling_order(1)%height_type
!       lake%current_filling_cell_index = 1
!       lake%next_cell_volume_threshold = &
!         lake_parameters%filling_order(1)%fill_threshold
!       lake%previous_cell_volume_threshold = 0.0_dp
!       lake%lake_volume = 0.0_dp
!     end if
! end subroutine initialise_filling_lake

! subroutine initialise_overflowing_lake(lake,
!                                        current_redirect,
!                                        excess_water)
!   type(redirect), pointer, intent(in) :: current_redirect
!   real(dp), intent(in) :: excess_water
!   type(lakeprognostics), pointer, intent(inout) :: lake
!     call set_filling_lake_parameter_to_null_values(lake)
!     call set_subsumed_lake_parameter_to_null_values(lake)
!     lake%current_redirect = current_redirect
!     lake%excess_water = excess_water
! end subroutine initialise_overflowing_lake

! subroutine initialise_subsumed_lake(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!     call set_filling_lake_parameter_to_null_values(lake)
!     call set_overflowing_lake_parameter_to_null_values(lake)
!     lake%redirect_target = -1
! end subroutine initialise_subsumed_lake

! !##############################################################################
! !##############################################################################
! !                  Section 3: Utility subprograms
! !##############################################################################
! !##############################################################################

! subroutine add_entry_to_dictionary(dict,key,value)
!   type(redirectdictionary) :: dict
!   integer :: key
!   type(redirect) :: value
!     if (dict%initialisation_complete) then
!       write(*,*) "Error - trying to write to finished dictionary"
!       stop
!     end if
!     dict%keys(dict%next_entry_index)
!     dict%values(dict%next_entry_index)
!     dict%next_entry_index = dict%next_entry_index + 1
! end subroutine add_entry_to_dictionary

! subroutine finish_dictionary(dict)
!   type(redirectdictionary), intent(in) :: dict
!   integer, dimension(:), allocatable :: sorted_keys
!   type(redirect), dimension(:), allocatable :: sorted_values
!   integer :: minimum_location
!   integer :: number_of_entries
!   integer :: i
!     dict%number_of_entries = size(dict%keys)
!     max_key_value = max(dict%keys)
!     allocate(sorted_keys(dict%number_of_entries)
!     allocate(sorted_values(dict%number_of_entries)
!     do i = 1,number_of_entries
!       minimum_location = minloc(dict%keys)
!       sorted_keys(i) = dict%keys(minimum_location)
!       sorted_values(i) = dict%values(minimum_location)
!       dict%keys(i) = max_key_value + 1
!     end do
!     dict%keys(:) = sorted_keys(:)
!     dict%values(:) = sorted_values(:)
!     dict%initialisation_complete = .true.
! end subroutine finish_dictionary

! function get_dictionary_entry(dict,key) result(value)
!   type(redirectdictionary), intent(in) :: dict
!   integer :: section_start_index
!   integer :: section_end_index
!   integer :: section_length
!   integer :: midpoint_index
!   integer :: midpoint_key_value
!     if (.not. dict%initialisation_complete) then
!       write(*,*) "Error - trying to read from unfinished dictionary"
!       stop
!     end if
!     section_start_index = 1
!     section_end_index = dict%number_of_entries
!     section_length  = dict%number_of_entries
!     do
!       if (section_length > 4) then
!         midpoint_index = section_length/2 + section_start_index
!         midpoint_key_value = dict%keys(midpoint_index)
!         if (midpoint_key_value == key)
!             value = dict%values(midpoint_index)
!             return
!         else if (midpoint_key_value > key)
!           section_end_index = midpoint_index - 1
!           section_length  = section_end_index + 1 - section_start_index
!         else
!           section_start_index = midpoint_index + 1
!           section_length  = section_end_index + 1 - section_start_index
!         end if
!       else
!         do i = section_start_index,section_end_index
!           if (dict%keys(i) == key) then
!             value = dict%values(i)
!             return
!           end if
!         end do
!       end if
!     end do
! end function get_dictionary_entry

! !##############################################################################
! !##############################################################################
! !                  Section 4: Subprograms on individual lakes
! !##############################################################################
! !##############################################################################

! recursive subroutine add_water(lake,inflow,store_water)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   real(dp), intent(in) :: inflow
!   logical, intent(in) :: store_water
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   type(lakeprognostics), pointer :: sublake
!   type(lakeprognostics), pointer :: other_lake
!   _DEFINE_SURFACE_MODEL_COORDS_
!   real(dp) :: new_lake_volume
!   integer :: other_lake_number
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     if (lake%lake_type == filling_lake_type) then
!       if (.not. lake%active_lake) then
!         !Take the first sub-basin of this lake
!         sublake => lake_model_prognostics%lakes(lake_parameters%secondary_lakes(1))
!         call add_water(sublake,inflow,.false.)
!       end if
!       if (store_water) then
!         lake%unprocessed_water = lake%unprocessed_water + inflow
!       end if
!       do while (inflow > 0.0_dp)
!         if (lake%lake_volume >= 0 .and. lake%current_filling_cell_index == 1 .and.
!             lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) == 0 .and.
!             .not. lake_parameters%is_leaf .and. lake%current_height_type == flood_height) then
!           surface_model_coords = &
!             get_corresponding_surface_model_grid_cell(lake%current_cell_to_fill,
!                                                       lake_model_parameters%grid_specific_lake_model_parameters)
!           lake_model_prognostics%lake_cell_count(surface_model_coords) = &
!             lake_model_prognostics%lake_cell_count(surface_model_coords) + 1
!           lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) = &
!             lake%parameters%lake_number
!         end if
!         new_lake_volume = inflow + lake%lake_volume
!         if (new_lake_volume <= lake%next_cell_volume_threshold) then
!           lake%lake_volume = new_lake_volume
!           inflow = 0.0_dp
!         else
!           inflow = new_lake_volume - lake%next_cell_volume_threshold
!           lake%lake_volume = lake%next_cell_volume_threshold
!           if (lake%current_filling_cell_index == size(lake%parameters%filling_order)) then
!             if (check_if_merge_is_possible(lake_parameters,lake_model_prognostics)) then
!               call merge_lakes(inflow,lake_parameters,
!                                lake_model_prognostics)
!             else
!               call change_to_overflowing_lake(lake,inflow)
!             end if
!           end if
!           lake%current_filling_cell_index = lake%current_filling_cell_index + 1
!           lake%previous_cell_volume_threshold = lake%next_cell_volume_threshold
!           lake%next_cell_volume_threshold = &
!             lake%parameters%filling_order(lake%current_filling_cell_index)%fill_threshold
!           lake%current_cell_to_fill = &
!             lake%parameters%filling_order(lake%current_filling_cell_index)%coords
!           lake%current_height_type = &
!             lake%parameters%filling_order(lake%current_filling_cell_index)%height_type
!           if (lake%current_height_type == flood_height .and. &
!               lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) == 0) then
!             surface_model_coords = &
!               get_corresponding_surface_model_grid_cell(lake%current_cell_to_fill,
!                                                         lake_model_parameters%grid_specific_lake_model_parameters)
!             lake_model_prognostics%lake_cell_count(surface_model_coords) = &
!               lake_model_prognostics%lake_cell_count(surface_model_coords) + 1
!             lake_model_prognostics%lake_numbers,lake(current_cell_to_fill) = &
!               lake%parameters%lake_number
!           end if
!         end if
!       end do
!     else if (lake%lake_type == overflowing_lake_type) then
!       if (.not. lake%active_lake) then
!         write(*,*) "Water added to inactive lake"
!         stop
!       end if
!       if (store_water) then
!         lake%unprocessed_water = lake%unprocessed_water + inflow
!       end if
!       if (lake%current_redirect%use_local_redirect) then
!         other_lake_number = lake%current_redirect%local_redirect_target_lake_number
!         other_lake => &
!           lake_model_prognostics%lakes(other_lake_number)
!         call add_water(other_lake,inflow,.false.)
!       else
!         lake%excess_water = lake%excess_water + inflow
!       end if
!     else if (lake%lake_type == subsumed_lake_type) then
!       if (.not. lake%active_lake) then
!         write(*,*) "Water added to inactive lake"
!         stop
!       end if
!       if (lake%redirect_target == -1) then
!         lake%redirect_target = find_root(lake%lake_model_prognostics%set_forest, &
!                                          lake_parameters%lake_number)
!       end if
!       call add_water(lake_model_prognostics%lakes(lake%redirect_target), &
!                      inflow,store_water)
!     else
!       write(*,*) "Unrecognised lake type"
!       stop
!     end if
! end subroutine add_water

! recursive subroutine remove_water(lake,outflow,store_water)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   real(dp), intent(in) :: outflow
!   logical, intent(in) :: store_water
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   _DEFINE_SURFACE_MODEL_COORDS_
!   real(dp) :: new_lake_volume
!   real(dp) :: minimum_new_lake_volume
!   integer :: redirect_target
!   logical :: repeat_loop
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     if (lake%lake_type == filling_lake_type) then
!       if (.not. lake%active_lake) then
!         write(*,*) "Water removed from inactive lake"
!         stop
!       end if
!       if (store_water) then
!         lake%unprocessed_water = lake%unprocessed_water - outflow
!       end if
!       repeat_loop = .false.
!       do while (outflow > 0.0_dp .or. repeat_loop)
!         repeat_loop = .false.
!         new_lake_volume = lake%lake_volume - outflow
!         if (new_lake_volume > 0 .and. new_lake_volume <= &
!             lake_model_parameters%lake_model_settings%minimum_lake_volume_threshold .and. &
!             lake_parameters%is_leaf) then
!           lake_model_prognostics%lake_water_from_ocean(lake_parameters%center_cell_coarse_coords) = &
!               lake_model_prognostics%lake_water_from_ocean(
!                 lake_parameters%center_cell_coarse_coords) - new_lake_volume)
!           new_lake_volume = 0
!         end if
!         minimum_new_lake_volume = lake%previous_cell_volume_threshold
!         if (new_lake_volume <= 0.0_dp .and. lake%current_filling_cell_index == 1) then
!           outflow = 0.0_dp
!           if (lake_parameters%is_leaf) then
!             lake%lake_volume = new_lake_volume
!           else
!             lake%lake_volume = 0.0_dp
!             call split_lake(lake,-new_lake_volume,lake_parameters,
!                             lake_model_prognostics)
!           end if
!           if (lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) == &
!               lake%parameters%lake_number .and. &
!               .not. lake_parameters%is_leaf .and. lake%current_height_type == flood_height) then
!             surface_model_coords = &
!               get_corresponding_surface_model_grid_cell(lake%current_cell_to_fill,
!                                                         lake_model_parameters%grid_specific_lake_model_parameters)
!             lake_model_prognostics%lake_cell_count(surface_model_coords) = &
!               lake_model_prognostics%lake_cell_count(surface_model_coords) - 1
!             lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) = 0
!           end if
!         else if (new_lake_volume >= minimum_new_lake_volume .and. &
!                  new_lake_volume > 0) then
!           outflow = 0.0_dp
!           lake%lake_volume = new_lake_volume
!         else
!           outflow = minimum_new_lake_volume - new_lake_volume
!           lake%lake_volume = minimum_new_lake_volume
!           if (lake%current_height_type == flood_height .and. &
!               lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) == &
!               lake%parameters%lake_number) then
!             surface_model_coords = &
!               get_corresponding_surface_model_grid_cell(lake%current_cell_to_fill,
!                                                         lake_model_parameters%grid_specific_lake_model_parameters)
!             lake_model_prognostics%lake_cell_count(surface_model_coords) = &
!               lake_model_prognostics%lake_cell_count(surface_model_coords) - 1
!             lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) = 0
!           end if
!           lake%current_filling_cell_index = lake%current_filling_cell_index - 1
!           lake%next_cell_volume_threshold = lake%previous_cell_volume_threshold
!           if (lake%current_filling_cell_index > 1) then
!             lake%previous_cell_volume_threshold = &
!               lake%parameters%filling_order(lake%current_filling_cell_index-1)%fill_threshold
!           else
!             lake%previous_cell_volume_threshold = 0.0_dp
!           end if
!           lake%current_cell_to_fill = lake%parameters%filling_order(lake%current_filling_cell_index)%coords
!           lake%current_height_type = &
!             lake%parameters%filling_order(lake%current_filling_cell_index)%height_type
!           if (lake%current_filling_cell_index > 1) then
!             repeat_loop = .true.
!           end if
!         end if
!       end do
!     else if (lake%lake_type == overflowing_lake_type) then
!       if (.not. lake%active_lake) then
!         write(*,*) "Water removed from inactive lake"
!         stop
!       end if
!       if (store_water) then
!         lake%unprocessed_water = lake%unprocessed_water - outflow
!       end if
!       if (outflow <= lake%excess_water) then
!         lake%excess_water = lake%excess_water - outflow
!       end if
!       outflow = outflow - lake%excess_water
!       lake%excess_water = 0.0_dp
!       call change_overflowing_lake_to_filling_lake(lake)
!       call remove_water(lake,outflow,.false.)
!     else if (lake%lake_type == subsumed_lake_type) then
!       !Redirect to primary
!       if (.not. lake%active_lake) then
!         write(*,*) "Water removed from inactive lake"
!         stop
!       end if
!       if (lake%redirect_target == -1) then
!         lake%redirect_target = find_root(lake_model_prognostics%set_forest, &
!                                          lake_parameters%lake_number)
!       end if
!       !Copy this variables from the lake object as the lake object will
!       !have its redirect set to -1 if the redirect target splits
!       redirect_target = lake%redirect_target
!       call remove_water(lake_model_prognostics%lakes(redirect_target),
!                         outflow,store_water)
!     else
!       write(*,*) "Unrecognised lake type"
!       stop
!     end if
! end subroutine remove_water

! subroutine process_water(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!     lake%unprocessed_water = 0.0_dp
!     if (lake%unprocessed_water > 0.0_dp) then
!         call add_water(lake,lake%unprocessed_water,.false.))
!     else if (lake%unprocessed_water < 0.0_dp) then
!         call remove_water(lake,-lake%unprocessed_water,.false.))
!     end if
! end subroutine process_water

! function check_if_merge_is_possible(lake_parameters,
!                                     lake_model_prognostics) result(merge_possible)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   logical :: all_secondary_lakes_filled
!   integer :: secondary_lake
!   logical :: merge_possible
!     if (.not. lake_parameters%is_primary) then
!       all_secondary_lakes_filled = .true.
!       do i = 1,size(lake_model_prognostics%lakes(lake_parameters%primary_lake)% &
!                    parameters%secondary_lakes)
!         secondary_lake => lake_model_prognostics%lakes(lake_parameters%primary_lake)%
!                             parameters%secondary_lakes(i)
!         if (secondary_lake /= lake_parameters%lake_number) then
!           all_secondary_lakes_filled = all_secondary_lakes_filled .and. &
!             (lake_model_prognostics%lakes(secondary_lake)%lake_type == overflowing_lake_type)
!         end if
!       end do
!       merge_possible = all_secondary_lakes_filled
!     else
!       merge_possible = .false.
!     end if
! end function check_if_merge_is_possible

! subroutine merge_lakes(inflow,
!                        lake_parameters,
!                        lake_model_prognostics)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   real(dp), intent(in) :: inflow
!   type(lakeprognostics), pointer, intent(inout) :: other_lake
!   real(dp) :: excess_water
!   real(dp) :: total_excess_water
!   integer :: secondary_lake
!     total_excess_water = 0.0_dp
!     primary_lake => lake_model_prognostics%lakes(lake_parameters%primary_lake)
!     primary_lake%variables%active_lake = .true.
!     do i = 1,size(primary_lake%parameters%secondary_lakes)
!       secondary_lake = primary_lake%parameters%secondary_lakes(i)
!       other_lake = lake_model_prognostics%lakes(secondary_lake)
!       if (((other_lake%lake_type == filling_lake_type) .and. &
!             lake_parameters%lake_number /= secondary_lake) .or. &
!            (other_lake%lake_type == subsumed_lake_type)) then
!         write(*,*) "wrong lake type when merging"
!         stop
!       end if
!       call change_to_subsumed_lake(other_lake,excess_water)
!       total_excess_water = total_excess_water + excess_water
!     end do
!       call add_water(lake_model_prognostics%lakes(primary_lake%parameters%lake_number),
!                      inflow+total_excess_water,.false.)
! end subroutine merge_lakes

! subroutine split_lake(lake,water_deficit,
!                       lake_parameters,
!                       lake_model_prognostics)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   type(lakeparameters), intent(in) :: lake_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   real(dp), intent(in) :: water_deficit
!   real(dp) :: water_deficit_per_lake
!   integer :: secondary_lake
!     lake%active_lake = .false.
!     water_deficit_per_lake = water_deficit/size(lake_parameters%secondary_lakes)
!     do i = 1,size(lake_parameters%secondary_lakes)
!       secondary_lake = lake_parameters%secondary_lakes(i)
!       call change_subsumed_to_filling_lake(lake_model_prognostics%lakes(secondary_lake), &
!                                            lake_parameters%lake_number)
!       call remove_water(lake_model_prognostics%lakes(secondary_lake), &
!                         water_deficit_per_lake,.false.)
!     end do
! end subroutine split_lake

! subroutine change_to_overflowing_lake(lake,inflow)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   real(dp), intent(in) :: inflow
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   type(redirect) :: working_redirect
!   integer :: secondary_lake
!   logical :: redirect_target_found
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     ! if debug
!     !   println("Lake $(lake_parameters%lake_number) changing from filling to overflowing lake ")
!     ! end
!     if (.not. lake_parameters%is_primary) then
!       do i = 1,size(lake_model_prognostics%lakes(lake_parameters%primary_lake)% &
!                     parameters%secondary_lakes)
!         secondary_lake => lake_model_prognostics%lakes(lake_parameters%primary_lake)% &
!                           parameters%secondary_lakes(i)
!         redirect_target_found = .false.
!         if ((lake_model_prognostics%lakes(secondary_lake)%lake_type == filling_lake_type) .and.
!             (secondary_lake /= lake%parameters%lake_number)) then
!           working_redirect = &
!             lake_parameters%outflow_points(secondary_lake)
!           redirect_target_found = .true.
!           exit
!         end if
!       end do
!       if (.not. redirect_target_found) then
!         write(*,*) "Non primary lake has no valid outflows"
!         stop
!       end if
!     else
!       NEEDS WORK
!       if (haskey(lake_parameters%outflow_points,-1)) then
!         working_redirect = lake_parameters%outflow_points(-1)
!       else
!         do for key in keys(lake_parameters%outflow_points)
!           working_redirect = lake_parameters%outflow_points(key)
!         end do
!       end if
!       NEEDS WORK UP TO HERE
!     end if
!     call initialise_overflowing_lake(lake,
!                                      working_redirect,
!                                      inflow)
! end subroutine change_to_overflowing_lake

! subroutine change_to_subsumed_lake(lake,excess_water)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   real(dp),intent(out) :: excess_water
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     ! if debug
!     !   println("Lake $(lake_parameters%lake_number) accepting merge")
!     ! end
!     call initialise_subsumed_lake(lake)
!     call make_new_link(lake_model_prognostics%set_forest, &
!                        lake_parameters%primary_lake, &
!                        lake_parameters%lake_number)
!     if (lake%lake_type == overflowing_lake_type) then
!       excess_water = lake%excess_water
!     else
!       excess_water = 0.0_dp
!     end if
! end subroutine change_to_subsumed_lake

! subroutine change_overflowing_lake_to_filling_lake(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     if (lake%excess_water /= 0.0_dp) then
!       write(*,*) "Can't change lake with excess water back to filling lake"
!       stop
!     end if
!     ! if debug
!     !   println("Lake $(lake_parameters%lake_number) changing from overflowing to filling lake ")
!     ! end
!     call initialise_filling_lake(lake,.true.)
! end subroutine change_overflowing_lake_to_filling_lake

! subroutine change_subsumed_to_filling_lake(lake,split_from_lake_number)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   integer, intent(in) :: split_from_lake_number
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     call split_set(lake_model_prognostics%set_forest, &
!                    split_from_lake_number, &
!                    lake_parameters%lake_number)
!     NEEDS WORK
!     for_elements_in_set(lake_model_prognostics%set_forest,
!                         lake_parameters%lake_number,
!                         x ->
!                         lake_model_prognostics%lakes(get_label(x))%redirect_target = -1)
!     NEEDS WORK UP TO HERE
!     call initialise_filling_lake(lake,.true.)
! end subroutine change_subsumed_to_filling_lake

! subroutine drain_excess_water(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   type(lakeprognostics), pointer :: other_lake
!   real(dp) :: total_lake_volume
!   real(dp) :: excess_water
!   integer :: other_lake_number
!     lake_parameters => lake%parameters
!     lake_model_parameters => lake%lake_model_parameter
!     lake_model_prognostics => lake%lake_model_prognostics
!     if (lake%excess_water > 0.0_dp) then
!       if (lake%current_redirect%use_local_redirect) then
!         other_lake_number = lake%current_redirect%local_redirect_target_lake_number
!         other_lake => &
!           lake_model_prognostics%lakes(other_lake_number)
!         excess_water = lake%excess_water
!         lake%excess_water = 0.0_dp
!         call add_water(other_lake,excess_water,.false.)
!       else
!         total_lake_volume = 0.0_dp
!         NEEDS WORK
!         for_elements_in_set(lake_model_prognostics%set_forest,
!                             find_root(lake_model_prognostics%set_forest,
!                                       lake_parameters%lake_number),
!                             x -> total_lake_volume = total_lake_volume +
!                             get_lake_volume(lake_model_prognostics%lakes(get_label(x))))
!         NEEDS WORK UP TO HERE
!         flow = (total_lake_volume)/ &
!                (lake_model_parameters%lake_model_settings%lake_retention_constant + 1.0_dp)
!         flow = min(flow,lake%excess_water)
!         lake_model_prognostics%water_to_hd( &
!           lake%current_redirect%non_local_redirect_target) = &
!              lake_model_prognostics%water_to_hd(lake%current_redirect% &
!                                                 non_local_redirect_target)+flow
!         lake%excess_water = lake%excess_water - flow
!       end if
!     end if
! end subroutine drain_excess_water

! subroutine release_negative_water(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     lake_parameters => lake%parameters
!     lake_model_prognostics => lake%lake_model_prognostics
!     lake_model_prognostics%lake_water_from_ocean( &
!               lake_parameters%center_cell_coarse_coords) = &
!       lake_model_prognostics%lake_water_from_ocean(lake_parameters%center_cell_coarse_coords) -
!         lake%lake_volume
!     lake%lake_volume = 0.0_dp
! end subroutine release_negative_water

! function get_lake_volume(lake) result(lake_volume)
!   type(lakeprognostics), pointer, intent(in) :: lake
!   real(dp) :: lake_volume
!   NEEDS WORK
!   lake_volume = get_lake_volume(lake::FillingLake) = lake%lake_volume + lake%variables%unprocessed_water

!   lake_volume = get_lake_volume(lake::OverflowingLake) = lake%parameters%filling_order(end)%fill_threshold + &
!                                          lake%excess_water +
!                                          lake%variables%unprocessed_water

!   lake_volume = get_lake_volume(lake::SubsumedLake) = lake%parameters%filling_order(end)%fill_threshold + &
!                                                       lake%variables%unprocessed_water
!   NEEDS WORK UP TO HERE
! end function get_lake_volume

! function get_lake_filled_cells(

!   NEEDS WORK
!   get_lake_filled_cells(lake::FillingLake) =
!   map(f->f%coords,lake%parameters%
!                   filling_order(1:lake%current_filling_cell_index))

! get_lake_filled_cells(lake::Union{OverflowingLake,SubsumedLake}) =
!   map(f->f%coords,lake%parameters%filling_order)
!   NEEDS WORK UP TO HERE
! end function get_lake_filled_cells

! subroutine calculate_effective_lake_volume_per_cell(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   type(lakeparameters), pointer :: lake_parameters
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   real(dp) :: total_lake_volume
!   real(dp) :: effective_volume_per_cell
!   integer :: total_number_of_flooded_cells
!     if (lake%lake_type == filling_lake_type .and.
!         lake%lake_type == overflowing_lake_type) then
!       lake_parameters => lake%parameters
!       lake_model_parameters => lake%lake_model_parameter
!       lake_model_prognostics => lake%lake_model_prognostics
!       total_number_of_flooded_cells = 0
!       total_lake_volume = 0.0_dp
!       NEEDS WORK
!       working_cell_list::Vector{CartesianIndex} = CartesianIndex[]
!       for_elements_in_set(lake_model_prognostics%set_forest,
!                           find_root(lake_model_prognostics%set_forest,
!                                     lake%parameters%lake_number),
!                           function (x)
!                             other_lake::Lake = lake_model_prognostics%lakes(get_label(x))
!                             total_lake_volume = total_lake_volume + get_lake_volume(other_lake)
!                             other_lake_working_cells::Vector{CartesianIndex} =
!                               get_lake_filled_cells(other_lake)
!                             total_number_of_flooded_cells = total_number_of_flooded_cells +
!                               size(other_lake_working_cells)
!                             append!(working_cell_list,
!                                     other_lake_working_cells)
!                           end)
!       NEEDS WORK UP TO HERE
!       effective_volume_per_cell = &
!         total_lake_volume / total_number_of_flooded_cells
!       NEEDS WORK
!       do for coords::CartesianIndex in working_cell_list
!         surface_model_coords = get_corresponding_surface_model_grid_cell(coords, &
!                                   lake_model_parameters%grid_specific_lake_model_parameters)
!         lake_model_prognostics% &
!           effective_volume_per_cell_on_surface_grid(surface_model_coords) = &
!             lake_model_prognostics% &
!             effective_volume_per_cell_on_surface_grid(surface_model_coords) + &
!             effective_volume_per_cell)
!       end do
!       NEEDS WORK UP TO HERE
!     else if (lake%lake_type == subsumed_lake_type) then
!       calculate_effective_lake_volume_per_cell::CalculateEffectiveLakeVolumePerCell)
!     else
!       write(*,*) "Unrecognised lake type"
!       stop
!     end if
! end subroutine calculate_effective_lake_volume_per_cell

! subroutine show(lake)
!   type(lakeprognostics), pointer, intent(inout) :: lake
!   type(lakeparameters), pointer :: lake_parameters
!     lake_parameters => lake%parameters
!     write(*,*) "-----------------------------------------------"
!     write(*,*) "Lake number: ", lake_parameters%lake_number
!     write(*,*) "Center cell: ", lake_parameters%center_coords
!     write(*,*) "Unprocessed water: ", lake_variables%unprocessed_water
!     write(*,*) "Active Lake: ", lake_variables%active_lake
!     write(*,*) "Center cell coarse coords: ", lake_parameters%center_cell_coarse_coords
!     if (lake%lake_type == overflowing_lake_type) then
!       write(*,*) "O"
!       write(*,*) "Excess water: ", lake%excess_water
!       write(*,*) "Current Redirect: ", lake%current_redirect
!     else if (lake%lake_type == filling_lake_type) then
!       write(*,*) "F"
!       write(*,*) "Lake volume: ", lake%lake_volume
!       write(*,*) "Current cell to fill: ", lake%current_cell_to_fill
!       write(*,*) "Current height type: ", lake%current_height_type
!       write(*,*) "Current filling cell index: ", lake%current_filling_cell_index
!       write(*,*) "Next cell volume threshold: ", lake%next_cell_volume_threshold
!       write(*,*) "Previous cell volume threshold: ", lake%previous_cell_volume_threshold
!     else if (lake%lake_type == subsumed_lake_type) then
!       write(*,*) "S"
!       write(*,*) "Redirect target : ", lake%redirect_target
!     end if
! end subroutine show

! !##############################################################################
! !##############################################################################
! !            Section 5: Subprograms on the full ensemble of lakes
! !##############################################################################
! !##############################################################################

! subroutine create_lakes(lake_model_parameters,
!                         lake_model_prognostics,
!                         lake_parameters_as_array::Array{Float64})
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   integer :: basin_number
!   logical :: contains_lake
!   logical :: basins_found
!     lake_parameters_array::Array{LakeParameters} =
!       get_lake_parameters_from_array(lake_parameters_as_array, &
!                                      lake_model_parameters%lake_model_grid, &
!                                      lake_model_parameters%hd_model_grid; &
!                                      single_index= &
!                                      isa(lake_model_parameters%lake_model_grid, &
!                                          UnstructuredGrid))
!     do i = 1,size(lake_parameters_array)
!       lake_parameters::LakeParameters = lake_parameters_array(i)
!       if (i /= lake_parameters%lake_number) then
!         write(*,*) "Lake number doesn't match position when creating lakes"
!         stop
!       end if
!       call add_set(lake_model_prognostics%set_forest,lake_parameters%lake_number)
!       lake::FillingLake = FillingLake(lake_parameters, &
!                                       LakeVariables(lake_parameters%is_leaf), &
!                                       lake_model_parameters, &
!                                       lake_model_prognostics, &
!                                       .false.)
!       lake_model_prognostics%lakes(i) => lake
!       if (lake_parameters%is_leaf .and. (lake%current_height_type == flood_height &
!                                   .or. size(lake_parameters%filling_order) == 1)) then
!         surface_model_coords = &
!           get_corresponding_surface_model_grid_cell(lake%current_cell_to_fill, &
!                                                     lake_model_parameters%grid_specific_lake_model_parameters)
!         lake_model_prognostics%lake_cell_count(surface_model_coords) = &
!           lake_model_prognostics%lake_cell_count(surface_model_coords) + 1
!         lake_model_prognostics%lake_numbers(lake%current_cell_to_fill) = &
!           lake%parameters%lake_number
!       end if
!     end do
!     do i=1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       lake_model_parameters%lake_centers(lake%parameters%center_coords) = .true.
!     end do
!     allocate(basins_in_coarse_cell_temp((this%nlat*this%nlon)&
!                                         /(this%nlat_coarse*this%nlon_coarse)))
!     allocate(this%basin_numbers(this%nlat_coarse,this%nlon_coarse))
!     allocate(basins_temp(this%nlat_coarse*this%nlon_coarse))
!     NEEDS CLEAN UP AND MAKING GENERIC
!     basin_number = 1
!     lat_scale_factor = nlat_in/nlat_coarse_in
!     lon_scale_factor = nlon_in/nlon_coarse_in
!     lake_center_count
!     _LOOP_OVER_HD_GRID_ _COORDS_HD_
!       number_of_basins_in_coarse_cell = 0
!       contains_lake = .false.
!       do l = 1+(j-1)*lon_scale_factor,j*lon_scale_factor
!           do k = 1+(i-1)*lat_scale_factor,i*lat_scale_factor
!       do for_all_fine_cells_in_coarse_cell(lake_model_parameters%lake_model_grid,
!                                         lake_model_parameters%hd_model_grid,
!                                         _COORDS_HD_) do fine_coords::CartesianIndex
!         if (lake_model_parameters%lake_centers(fine_coords)) then
!           contains_lake = .true.
!           do i=1,size(lake_model_prognostics%lakes)
!             lake => lake_model_prognostics%lakes(i)
!             if (lake%parameters%center_coords == fine_coords) then
!               number_of_basins_in_coarse_cell = number_of_basins_in_coarse_cell + 1
!               basins_in_coarse_cell_temp(number_of_basins_in_coarse_cell) = lake%parameters%lake_number
!             end if
!           end do
!         end if
!       end do
!       if (contains_lake ) then
!         INDICES_cells_with_lakes_temp_INDEX_NAME(basin_number) = _COORDS_HD_
!       end do
!       end do
!       if (basins_found) then
!         push!(lake_model_parameters%basins,basins_in_coarse_cell)
!         lake_model_parameters%basin_numbers(_COORDS_HD_) = basin_number
!         basin_number = basin_number + 1
!       end if
!       if(number_of_basins_in_coarse_cell > 0) then
!         allocate(basins_in_coarse_cell(number_of_basins_in_coarse_cell))
!         do m=1,number_of_basins_in_coarse_cell
!           basins_in_coarse_cell(m) = basins_in_coarse_cell_temp(m)
!         end do
!         basin_number = basin_number + 1
!         basins_temp(basin_number) = &
!           integerlist(basins_in_coarse_cell)
!         this%basin_numbers(i,j) = basin_number
!       end if
!     _LOOP_OVER_HD_GRID_END_
!     while
!       TRANSFER cells_with_lakes to shortened array
!     end do
!     deallocate(basins_temp)
!     deallocate(basins_in_coarse_cell_temp)
!     NEEDS CLEAN UP AND MAKING GENERIC RIGHT UP TO HERE
! end subroutine create_lakes

! subroutine setup_lakes(lake_model_parameters,lake_model_prognostics,&
!                        initial_water_to_lake_centers)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   real(dp) :: initial_water_to_lake_center
!   integer :: lake_index
!     _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
!       if (lake_model_parameters%lake_centers(_COORDS_LAKE_)) then
!         initial_water_to_lake_center = &
!           initial_water_to_lake_centers(_COORDS_LAKE_)
!         if (initial_water_to_lake_center > 0.0_dp) then
!           lake_index = lake_model_prognostics%lake_numbers(_COORDS_LAKE_)
!           lake => lake_model_prognostics%lakes(lake_index)
!           call add_water(lake,initial_water_to_lake_center,.false.)
!         end if
!       end if
!     _LOOP_OVER_LAKE_GRID_END_
! end subroutine setup_lakes

! subroutine distribute_spillover(lake_model_parameters,lake_model_prognostics,
!                               initial_spillover_to_rivers)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   real(dp) :: initial_spillover
!     _LOOP_OVER_HD_GRID_ _COORDS_HD_
!       initial_spillover = initial_spillover_to_rivers(_COORDS_HD_)
!       if (initial_spillover > 0.0_dp) then
!         lake_model_prognostics%water_to_hd(_COORDS_HD_) = &
!           lake_model_prognostics%water_to_hd(_COORDS_HD_) + initial_spillover
!       end if
!     end do
!     _LOOP_OVER_HD_GRID_END_
! end subroutine distribute_spillover

! subroutine run_lakes(lake_model_parameters,lake_model_prognostics)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!   real(dp) :: evaporation_per_lake_cell
!   real(dp) :: share_to_each_lake
!   real(dp) :: inflow_minus_evaporation
!   real(dp) :: evaporation
!   integer :: lake_cell_count
!   integer :: cell_count_check
!   integer :: map_index
!   integer :: target_lake_index
!   integer :: lake_index
!     lake_model_prognostics%water_to_hd(_DIMS_) = 0.0_dp
!     lake_model_prognostics%lake_water_from_ocean(_DIMS_) = 0.0_dp
!     lake_model_prognostics%evaporation_from_lakes(_DIMS_) = 0.0_dp
!     new_effective_volume_per_cell_on_surface_grid(_DIMS_) =
!       elementwise_multiple(lake_model_prognostics%effective_lake_height_on_surface_grid_to_lakes, &
!                            lake_model_parameters%cell_areas_on_surface_model_grid)
!     evaporation_on_surface_grid(_DIMS_) = &
!       lake_model_prognostics%effective_volume_per_cell_on_surface_grid - &
!                                                   new_effective_volume_per_cell_on_surface_grid
!     lake_model_prognostics%effective_volume_per_cell_on_surface_grid(_DIM_) = 0.0_dp
!     _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
!       lake_cell_count = lake_model_prognostics%lake_cell_count(_COORDS_SURFACE_)
!       if (evaporation_on_surface_grid(_COORDS_SURFACE_) /= 0.0_dp .and. lake_cell_count > 0) then
!         cell_count_check = 0
!         map_index = lake_model_parameters%surface_cell_to_fine_cell_map_numbers(_COORDS_SURFACE_)
!         evaporation_per_lake_cell = evaporation_on_surface_grid(_COORDS_SURFACE_)/lake_cell_count
!         if (map_index /= 0) then
!           do for fine_coords::CartesianIndex in lake_model_parameters%surface_cell_to_fine_cell_maps(map_index)
!             target_lake_index = lake_model_prognostics%lake_numbers(fine_coords)
!             if (target_lake_index /= 0) then
!               lake => lake_model_prognostics%lakes(target_lake_index)
!               lake_model_prognostics%evaporation_from_lakes(target_lake_index) = &
!                 lake_model_prognostics%evaporation_from_lakes(target_lake_index) + evaporation_per_lake_cell
!               cell_count_check = cell_count_check + 1
!             end if
!           end do
!         end if
!         if (cell_count_check /= lake_cell_count)
!           write(*,*) "Inconsistent cell count when assigning evaporation"
!           stop
!         end if
!       end if
!     _LOOP_OVER_SURFACE_GRID_END_
!     lake_model_prognostics%evaporation_applied(DIM) = .false.
!     do for coords::CartesianIndex in lake_model_parameters%cells_with_lakes
!       if (lake_model_prognostics%water_to_lakes(coords) > 0.0_dp) then
!         lakes_in_cell = &
!           lake_model_parameters%basins(lake_model_parameters%basin_numbers(coords))
!         filter!(lake_index::Int64->
!                 lake_model_prognostics%lakes(lake_index)%variables%active_lake,
!                 lakes_in_cell)
!         share_to_each_lake = lake_model_prognostics%water_to_lakes(coords)/size(lakes_in_cell)
!         do for lake_index in lakes_in_cell
!           lake => lake_model_prognostics%lakes(lake_index)
!           if (.not. lake_model_prognostics%evaporation_applied(lake_index)) then
!             inflow_minus_evaporation = share_to_each_lake - &
!                                        lake_model_prognostics%evaporation_from_lakes(lake_index)
!             if (inflow_minus_evaporation >= 0.0_dp) then
!               call add_water(lake,inflow_minus_evaporation,.true.)
!             else
!               call remove_water(lake,-1.0_dp*inflow_minus_evaporation,.true.)
!             end if
!             lake_model_prognostics%evaporation_applied(lake_index) = .true.
!           else
!             call add_water(lake,share_to_each_lake,.true.)
!           end if
!         end do
!       else if (lake_model_prognostics%water_to_lakes(coords) < 0.0_dp) then
!         lakes_in_cell = &
!           lake_model_parameters%basins(lake_model_parameters%basin_numbers(coords))
!         filter!(lake_index::Int64->
!                 lake_model_prognostics%lakes(lake_index)%variables%active_lake,
!                 lakes_in_cell)
!         share_to_each_lake = -1.0_dp*lake_model_prognostics%water_to_lakes(coords)/size(lakes_in_cell)
!         do for lake_index in lakes_in_cell
!           lake::Lake = lake_model_prognostics%lakes(lake_index)
!           call remove_water(lake,share_to_each_lake,.true.)
!         end do
!       end if
!     end do
!     do i = 1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       lake_index = lake%parameters%lake_number
!       if (.not. lake_model_prognostics%evaporation_applied(lake_index)) then
!         evaporation = lake_model_prognostics%evaporation_from_lakes(lake_index)
!         if (evaporation > 0.0) then
!           call remove_water(lake,evaporation,.true.)
!         else if (evaporation < 0.0) then
!           call add_water(lake,-1.0_dp*evaporation,.true.)
!         end if
!         lake_model_prognostics%evaporation_applied(lake_index) = .true.
!       end if
!     end do
!     do i=1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       if (lake%unprocessed_water /= 0.0_dp) then
!         call process_water(lake)
!       end if
!     end do
!     do i=1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       if (lake%lake_type == filling_lake_type) then
!         if (lake%lake_volume < 0.0_dp) then
!           call release_negative_water(lake)
!         end if
!       end if
!     end do
!     do i=1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       if (lake%lake_type == overflowing_lake_type) then
!         call drain_excess_water(lake)
!       end if
!     end do
!     do i=1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       call calculate_effective_lake_volume_per_cell(lake)
!     end do
! end subroutine run_lakes

! subroutine set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters
!                                                               lake_model_prognostics
!                                                               effective_lake_height_on_surface_grid)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!     _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
!       lake_model_prognostics%effective_lake_height_on_surface_grid_to_lakes(_COORDS_SURFACE_) = &
!         effective_lake_height_on_surface_grid(_COORDS_SURFACE_)
!     _LOOP_OVER_SURFACE_GRID_END_
! end subroutine set_effective_lake_height_on_surface_grid_to_lakes

! subroutine calculate_lake_fraction_on_surface_grid(lake_model_parameters,
!                                                    lake_model_prognostics,
!                                                    lake_fraction_on_surface_grid)
!   real(dp), dimension(:,:), intent(inout) :: lake_fraction_on_surface_grid
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!     lake_fraction_on_surface_grid(_DIMS_) = real(lake_model_prognostics%lake_cell_count(_DIMS_))&
!                                          /real(lake_model_parameters%number_fine_grid_cells(_DIMS_))
! end subroutine calculate_lake_fraction_on_surface_grid

! subroutine calculate_effective_lake_height_on_surface_grid(lake_model_parameters,
!                                                          lake_model_prognostics)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!     lake_model_prognostics%effective_lake_height_on_surface_grid_from_lakes(_DIMS_) = &
!       lake_model_prognostics%effective_volume_per_cell_on_surface_grid(_DIMS_) / &
!       lake_model_parameters%cell_areas_on_surface_model_grid(_DIMS_)
! end subroutine calculate_effective_lake_height_on_surface_grid

! subroutine print_results(lake_model_parameters,lake_model_prognostics, &
!                          timestep)
!   type(lakemodelparameters), intent(in) :: lake_model_parameters
!   type(lakemodelprognostics), intent(inout) :: lake_model_prognostics
!     write(*,*) "Timestep: $(timestep)"
!     call print_river_results(prognostic_fields)
!     write(*,*) ""
!     write(*,*) "Water to HD"
!     write(*,*) lake_model_prognostics%water_to_hd
!     write(*,*) "Water to lakes"
!     write(*,*) lake_model_prognostics%water_to_lakes
!     do i = 1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       write(*,*) "Lake Center: $(lake%parameters%center_coords) "*
!                  "Lake Volume: $(get_lake_volume(lake))"
!     end do
!     write(*,*) ""
!     write(*,*) "Diagnostic Lake Volumes:"
!     call print_lake_types(lake_model_parameters%lake_model_grid,lake_model_prognostics)
!     write(*,*) ""
!     diagnostic_lake_volumes(_DIMS_) =
!       calculate_diagnostic_lake_volumes_field(lake_model_parameters,
!                                               lake_model_prognostics)
!     write(*,*) diagnostic_lake_volumes
!     return prognostic_fields
! end subroutine print_results

! subroutine print_selected_lakes(lake_model_prognostics, &
!                                 lakes_to_print)
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     do for lake_number in print_selected_lakes%lakes_to_print
!       if (size(lake_model_prognostics%lakes) >= lake_number) then
!         lake => lake_model_prognostics%lakes(lake_number)
!         call show(lake)
!       end if
!     end do
!     return prognostic_fields
! end subroutine print_selected_lakes

! subroutine print_lake_types(grid::LatLonGrid,
!                             lake_model_prognostics)
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     do for_all_with_line_breaks(grid) do coords::Coords
!       call print_lake_type(coords,lake_model_prognostics)
!     end do
! end subroutine

! subroutine print_lake_type(coords::Coords,
!                            lake_model_prognostics::LakeModelPrognostics)
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   integer :: lake_number
!     lake_number = lake_model_prognostics%lake_numbers(coords)
!     if (lake_number == 0) then
!       print("- ")
!     else
!       lake => lake_model_prognostics%lakes(lake_number)
!       if (lake%variables%active_lake) then
!         if (lake%lake_type == filling_lake_type) then
!           print("F ")
!         else if (lake%lake_type == overflowing_lake_type) then
!           print("O ")
!         else if (lake%lake_type == subsumed_lake_type) then
!           print("S ")
!         else
!           print("U ")
!         end if
!       else
!         print("- ")
!       end if
!     end if
! end subroutine print_lake_type

! subroutine print_section(lake_model_parameters,lake_model_prognostics)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     write(*,*) ""
!     call print_lake_types_section(lake_model_parameters%lake_model_grid,lake_model_prognostics)
! end subroutine print_section

! subroutine print_lake_types_section(grid::LatLonGrid,
!                                     lake_model_prognostics)
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     section_coords::LatLonSectionCoords = LatLonSectionCoords(225,295,530,580)
!     do for_section_with_line_breaks(grid,section_coords) do coords::Coords
!       call print_lake_type(coords,lake_model_prognostics)
!     end do
! end subroutine print_lake_types_section

! subroutine write_lake_numbers(lake_model_parameters,lake_model_prognostics,
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

! function calculate_diagnostic_lake_volumes_field(lake_model_parameters, &
!                                                  lake_model_prognostics)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   type(lakeprognostics), pointer :: lake
!   real(dp) :: total_lake_volume
!   integer :: i
!     lake_volumes_by_lake_number::Vector{Float64} = &
!       zeros(Float64,size(lake_model_prognostics%lakes))
!     diagnostic_lake_volumes(_DIMS_) = 0.0_dp
!     do i = 1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       if (lake%variables%active_lake) then
!         total_lake_volume = 0.0_dp
!         for_elements_in_set(lake_model_prognostics%set_forest,
!                             find_root(lake_model_prognostics%set_forest,
!                                       lake%parameters%lake_number),
!                             x -> total_lake_volume = total_lake_volume +
!                             get_lake_volume(lake_model_prognostics%lakes(get_label(x))))
!         lake_volumes_by_lake_number(i) = total_lake_volume
!       end if
!     end do
!     _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
!       lake_number = lake_model_prognostics%lake_numbers(_COORDS_LAKE_)
!       if (lake_number > 0) then
!         diagnostic_lake_volumes(_COORDS_LAKE_) = lake_volumes_by_lake_number(lake_number)
!       end if
!     _LOOP_OVER_LAKE_GRID_END_
!     return diagnostic_lake_volumes
! end function calculate_diagnostic_lake_volumes_field

! subroutine set_lake_evaporation_for_testing(lake_model_parameters,lake_model_prognostics, &
!                                 lake_evaporation)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!     old_effective_lake_height_on_surface_grid(_DIMS_) = &
!       calculate_effective_lake_height_on_surface_grid(lake_model_parameters,lake_model_prognostics)
!     effective_lake_height_on_surface_grid(_DIMS_) = &
!       old_effective_lake_height_on_surface_grid - &
!       elementwise_divide(set_lake_evaporation%lake_evaporation, &
!                          lake_model_parameters%cell_areas_on_surface_model_grid)
!     call set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters,lake_model_prognostics, &
!                                                             effective_lake_height_on_surface_grid)
! end subroutine set_lake_evaporation_for_testing

! subroutine set_lake_evaporation(lake_model_parameters,lake_model_prognostics, &
!                                 height_of_water_evaporated)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   real(dp) :: working_effective_lake_height_on_surface_grid
!   _DEF_LOOP_OVER_SURFACE_GRID_INDEX_VARIABLES_
!     old_effective_lake_height_on_surface_grid(_DIMS_) = &
!       calculate_effective_lake_height_on_surface_grid(lake_model_parameters, &
!                                                       lake_model_prognostics)
!     effective_lake_height_on_surface_grid(_DIMS_) = &
!       old_effective_lake_height_on_surface_grid
!     _LOOP_OVER_SURFACE_GRID_ _COORDS_SURFACE_
!       if (old_effective_lake_height_on_surface_grid(_COORDS_SURFACE_) > 0.0_dp) then
!         working_effective_lake_height_on_surface_grid = &
!           old_effective_lake_height_on_surface_grid(_COORDS_SURFACE_) - &
!           height_of_water_evaporated(_COORDS_SURFACE_)
!         if (working_effective_lake_height_on_surface_grid > 0.0_dp) then
!           effective_lake_height_on_surface_grid(_COORDS_SURFACE_) = &
!             working_effective_lake_height_on_surface_grid
!         else
!           effective_lake_height_on_surface_grid(_COORDS_SURFACE_) = 0.0_dp
!         end if
!       end if
!     end do
!     _LOOP_OVER_SURFACE_GRID_END_
!     call set_effective_lake_height_on_surface_grid_to_lakes(&
!               lake_model_parameters,lake_model_prognostics, &
!               effective_lake_height_on_surface_grid)
! end subroutine set_lake_evaporation

! subroutine check_water_budget(lake_model_prognostics,
!                               lake_model_diagnostics,
!                               total_initial_water_volume)
!   type(lakemodelparameters), pointer :: lake_model_parameters
!   type(lakemodelprognostics), pointer :: lake_model_prognostics
!   real(dp), intent(in) :: total_initial_water_volume
!   real(dp) :: new_total_lake_volume
!   real(dp) :: change_in_total_lake_volume
!   real(dp) :: total_water_to_lakes
!   real(dp) :: total_inflow_minus_outflow
!   real(dp) :: difference
!   real(dp) :: tolerance
!   integer :: i
!     new_total_lake_volume = 0.0_dp
!     do i = 1,size(lake_model_prognostics%lakes)
!       lake => lake_model_prognostics%lakes(i)
!       new_total_lake_volume = new_total_lake_volume + get_lake_volume(lake)
!     end do
!     change_in_total_lake_volume = new_total_lake_volume - &
!                                   lake_model_diagnostics%total_lake_volume - &
!                                   total_initial_water_volume
!     total_water_to_lakes = sum(lake_model_prognostics%water_to_lakes)
!     total_inflow_minus_outflow = total_water_to_lakes + &
!                                  sum(lake_model_prognostics%lake_water_from_ocean) - &
!                                  sum(lake_model_prognostics%water_to_hd) - &
!                                  sum(lake_model_prognostics%evaporation_from_lakes)
!     difference = change_in_total_lake_volume - total_inflow_minus_outflow
!     tolerance = 10e-14*max(new_total_lake_volume, total_water_to_lakes)
!     if (abs(difference) >= tolerance)) then
!       write(*,*) "*** Lake Water Budget ***"
!       write(*,*) "Total lake volume: ",new_total_lake_volume
!       write(*,*) "Total inflow - outflow: ", total_inflow_minus_outflow
!       write(*,*) "Change in lake volume:", change_in_total_lake_volume
!       write(*,*) "Difference: ", difference
!       write(*,*) "Total water to lakes: " total_water_to_lakes
!       write(*,*) "Total water from ocean: ", sum(lake_model_prognostics%lake_water_from_ocean)
!       write(*,*) "Total water to HD model from lakes: ", sum(lake_model_prognostics%water_to_hd)
!       write(*,*) "Old total lake volume: ", lake_model_diagnostics%total_lake_volume
!     end if
!     lake_model_diagnostics%total_lake_volume = new_total_lake_volume
! end subroutine check_water_budget

end module l2_lake_model_mod
